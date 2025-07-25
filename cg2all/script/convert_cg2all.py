#!/usr/bin/env python

import os
import sys
import json
import time
import tqdm
import pathlib
import argparse
import tempfile
import shutil
import gc
import numpy as np
import torch
import dgl
import ctypes
import ctypes.util

os.environ["OPENMM_PLUGIN_DIR"] = "/dev/null"
import mdtraj

from cg2all.lib.libconfig import MODEL_HOME
from cg2all.lib.libdata import (
    PredictionData,
    create_trajectory_from_batch,
    create_topology_from_data,
    standardize_atom_name,
)
from cg2all.lib.residue_constants import read_coarse_grained_topology
import cg2all.lib.libcg
from cg2all.lib.libpdb import write_SSBOND
from cg2all.lib.libter import patch_termini
import cg2all.lib.libmodel

import warnings
warnings.filterwarnings("ignore")


def split_trajectory_into_chunks(pdb_fn, dcd_fn, chunk_size, temp_dir, output_basename,dcd_last,step_size=1):
    """
    Split a large trajectory into smaller chunks and save as temporary files.
    
    Args:
        pdb_fn: Path to PDB file
        dcd_fn: Path to DCD trajectory file
        chunk_size: Number of frames per chunk
        temp_dir: Directory to store temporary chunk files
        output_basename: Base name for output files (used in chunk naming)
    
    Returns:
        List of (pdb_fn, chunk_dcd_fn) tuples for each chunk
    """
    print(f"Loading trajectory to determine total frames...")
    
    # Load full trajectory to get frame count and metadata
    with mdtraj.open(dcd_fn) as f:
            total_frames_original = len(f)
    if int(dcd_last) == -1:
           total_frames = total_frames_original
    else:
           total_frames = int(dcd_last)

    total_frames_after_step = (total_frames + step_size -1)//step_size

    print(f"Total frames: {total_frames}. After skipping {step_size} frames: {total_frames_after_step}")
    print(f"Splitting into chunks of {chunk_size} frames...")
    
    chunk_files = []
    chunk_idx   = 0 
    
    for chunk_traj in mdtraj.iterload(dcd_fn, top=pdb_fn, chunk=chunk_size):
        if chunk_idx * chunk_size >= total_frames:
                  break

        if step_size > 1:
              chunk_traj = chunk_traj[::step_size]
        if len(chunk_traj) ==0:
              chunk_idx += 1
              continue

        print(f"Processing chunk {chunk_idx + 1} : {len(chunk_traj)} frames")

        #Save chunk as temporary dcd file
        chunk_dcd_fn = os.path.join(temp_dir, f"{output_basename}_input_chunk_{chunk_idx:04d}.dcd")
        chunk_traj.save_dcd(chunk_dcd_fn)


        chunk_files.append((pdb_fn, chunk_dcd_fn))
        del chunk_traj
        gc.collect()
        chunk_idx +=1

    return chunk_files





def process_single_chunk(chunk_pdb_fn, chunk_dcd_fn, model, cg_model, config, 
                        topology_map, device, batch_size, n_proc, chain_break_cutoff, 
                        is_all, fix_atom, standard_names, temp_dir, chunk_idx, output_basename,output_path,
                        save_reference_pdb=False):
    """
    Process a single trajectory chunk and save the output.
    
    Args:
        output_basename: Base name for output files (used in chunk naming)
        save_reference_pdb: If True, save the first frame as reference PDB
    
    Returns:
        Path to the output trajectory chunk file, and optionally reference PDB path
    """
    print(f"Processing chunk {chunk_idx}...")
    

    ##debugging helper function
    log_memory(f"chunk_{chunk_idx}_start")
    
    if torch.cuda.is_available():
          torch.cuda.empty_cache()

    # Create PredictionData for this chunk
    input_chunk = PredictionData(
        chunk_pdb_fn,
        cg_model,
        topology_map=topology_map,
        dcd_fn=chunk_dcd_fn,
        radius=config.globals.radius,
        chain_break_cutoff=0.1 * chain_break_cutoff,
        is_all=is_all,
        fix_atom=config.globals.fix_atom,
        batch_size=batch_size,
    )
    
    # Setup data loader
    if len(input_chunk) > 1 and (n_proc > 1 or batch_size > 1):
        input_loader = dgl.dataloading.GraphDataLoader(
            input_chunk, batch_size=batch_size, num_workers=n_proc, shuffle=False
        )
    else:
        input_loader = dgl.dataloading.GraphDataLoader(
            input_chunk, batch_size=1, num_workers=1, shuffle=False
        )
    
    # Process frames in this chunk
    xyz_list = []
    for batch in tqdm.tqdm(input_loader, desc=f"Chunk {chunk_idx}", leave=False):
        batch = batch.to(device)
        
        with torch.no_grad():
            R = model.forward(batch)[0]["R"].cpu().detach().numpy()
            mask = batch.ndata["output_atom_mask"].cpu().detach().numpy()

            coords = R[mask>0.0]
            xyz_list.append(coords)
      
            del R, mask, coords
        gc.collect() 
        
    
 
    # Combine xyz coordinates for this chunk
    if batch_size > 1:
        batch = dgl.unbatch(batch)[0]
        xyz_combined = np.concatenate(xyz_list, axis=0)
        del xyz_list
        gc.collect()

        n_frames_chunk = input_chunk.n_frame0 # Get actual number of frames in chunk
        xyz_chunk = xyz_combined.reshape((n_frames_chunk, -1, 3))
        del xyz_combined
        gc.collect()
    else:
        xyz_chunk = np.array(xyz_list)
        del xyz_list
        gc.collect()
    
    # Create topology and reorder atoms
    top, atom_index = create_topology_from_data(batch)

    xyz_reordered = xyz_chunk[:, atom_index]
    del xyz_chunk
    xyz_chunk = xyz_reordered
    del xyz_reordered
    gc.collect() 
    
    # Create trajectory for this chunk
    chunk_traj = mdtraj.Trajectory(
        xyz=xyz_chunk,
        topology=top,
        unitcell_lengths=input_chunk.cg.unitcell_lengths,
        unitcell_angles=input_chunk.cg.unitcell_angles,
    )
    log_memory(f"chunk_{chunk_idx}_processed")
    del  batch, top, atom_index, xyz_chunk
    gc.collect()
    # Apply post-processing
    output_chunk = patch_termini(chunk_traj)
  
    del chunk_traj
    gc.collect() 
    
    if standard_names:
        standardize_atom_name(output_chunk)
    
    reference_pdb_path = None   
    # Save reference PDB from first chunk if requested
    if save_reference_pdb:
        reference_pdb_path = os.path.join(output_path,f'{output_basename}.pdb')
        output_chunk[0].save_pdb(reference_pdb_path)
        print(f"Reference topology PDB created: {reference_pdb_path}")
        print(f"Reference topology has {output_chunk.n_atoms} atoms")    
    # Save chunk output
    #output_chunk_fn = os.path.join(temp_dir, f"{output_basename}_output_chunk_{chunk_idx:04d}.dcd")
    
#    xyz_data = output_chunk.xyz
    
#    cell_lengths = output_chunk.unitcell_lengths
#    cell_angles = output_chunk.unitcell_angles
    
    #with mdtraj.formats.DCDTrajectoryFile(output_chunk_fn,'w') as dcd_writer:
    #     dcd_writer.write(xyz_data,cell_lengths=cell_lengths,cell_angles=cell_angles)


    log_memory(f"chunk_{chunk_idx}_saved")

    del input_chunk    
    gc.collect()
#    drop_file_cache(output_chunk_fn)    
    
   
    if save_reference_pdb:
        return output_chunk, reference_pdb_path
    else:
        return output_chunk


def combine_trajectory_chunks(chunk_output, final_output_fn, save_reference_pdb=False):
    """
    Combine all processed trajectory chunks into final output file.
    Uses the reference PDB for consistent topology.
    """
    print("Combining trajectory chunks...")
    total_frames = 0
    n_atoms     = None 
    log_memory(f"Before opening_outputfile") 
    

    #Use DCD writer for streaming
    with mdtraj.formats.DCDTrajectoryFile(final_output_fn,'w') as writer:
        for i,chunk_traj in enumerate(chunk_output):
            log_memory(f"Before loading chunk {i} to checkpoint")   
            if i == 0:
               n_atoms = chunk_traj.n_atoms
            log_memory(f"After loading, Before writing chunk {i} to checkpoint") 
            chunk_traj_xyz_A = chunk_traj.xyz*10 
            chunk_traj_unitcell_A = chunk_traj.unitcell_lengths*10
            writer.write(chunk_traj_xyz_A,
                         cell_lengths=chunk_traj_unitcell_A,
                         cell_angles = chunk_traj.unitcell_angles)
            
            total_frames += len(chunk_traj)
     
            log_memory(f"After writing chunk {i} to checkpoint")
            del chunk_traj,chunk_traj_xyz_A,chunk_traj_unitcell_A
            gc.collect()
            
            
#            drop_file_cache(final_output_fn)
            log_memory(f"After garbage colleciton for  {i} chunk iter")

    print(f"Final trajectory: {total_frames} frames, {n_atoms} atoms")
    print(f"Final trajectory saved: {final_output_fn}")

def combine_checkpoint(checkpoint_files,reference_pdb_path, final_output_fn):
    """
    Combine all processed trajectory chunks into final output file.
    Uses the reference PDB for consistent topology.
    """
    print("Combining trajectory chunks...")
    total_frames = 0
    n_atoms     = None 
   
    #Use DCD writer for streaming
    with mdtraj.formats.DCDTrajectoryFile(final_output_fn,'w') as writer:
        for i,chunk_file in enumerate(tqdm.tqdm(checkpoint_files, desc="Loading chunks")):
            log_memory(f"Before loading chunk {i} to checkpoint") 
#            chunk_traj = mdtraj.load_dcd(chunk_file, top=reference_pdb_path)

                        
            with mdtraj.formats.DCDTrajectoryFile(chunk_file,'r') as reader:
                xyz, cell_lengths, cell_angles= reader.read()
                log_memory(f"After loading, Before writing chunk {i} to checkpoint") 
                writer.write(xyz,
                         cell_lengths=cell_lengths,
                         cell_angles =cell_angles)
                del xyz,cell_lengths,cell_angles
                gc.collect() 
            total_frames += len(xyz)
     
            log_memory(f"After writing chunk {i} to checkpoint")
                        
#            drop_file_cache(chunk_file)
#            log_memory(f"After cleaning chunk {i} cache")
#            drop_file_cache(final_output_fn)
#            log_memory(f"After output cache for {i} chunk iter")

    print(f"Final trajectory: {total_frames} frames, {n_atoms} atoms")
    print(f"Final trajectory saved: {final_output_fn}")

def main():
    arg = argparse.ArgumentParser(prog="convert_cg2all_chunked")
    arg.add_argument("-p", "--pdb", dest="in_pdb_fn", required=True)
    arg.add_argument("-d", "--dcd", dest="in_dcd_fn", default=None)  # Changed from required=True to default=None
    arg.add_argument("-o", "--out", "--output", dest="out_fn", required=True)
    arg.add_argument("-opdb", dest="outpdb_fn")
    arg.add_argument("--chunk-size", dest="chunk_size", default=1000, type=int, 
                    help="Number of frames per chunk (default: 1000)")
    arg.add_argument("--last",dest="dcd_last",default="-1")
    arg.add_argument(
        "--cg", dest="cg_model", default="CalphaBasedModel",
        choices=["CalphaBasedModel", "CA", "ca", 
                "ResidueBasedModel", "RES", "res", 
                "Martini", "martini", "Martini2", "martini2", 
                "Martini3", "martini3", 
                "PRIMO", "primo", 
                "BB", "bb", "backbone", "Backbone", "BackboneModel", 
                "MC", "mc", "mainchain", "Mainchain", "MainchainModel",
                "CACM", "cacm", "CalphaCM", "CalphaCMModel",
                "CASC", "casc", "CalphaSC", "CalphaSCModel",
                "SC", "sc", "sidechain", "SidechainModel"]
    )
    arg.add_argument("--chain-break-cutoff", dest="chain_break_cutoff", default=10.0, type=float)
    arg.add_argument("-a", "--all", "--is_all", dest="is_all", default=False, action="store_true")
    arg.add_argument("--fix", "--fix_atom", dest="fix_atom", default=False, action="store_true")
    arg.add_argument("--standard-name", dest="standard_names", default=False, action="store_true")
    arg.add_argument("--ckpt", dest="ckpt_fn", default=None)
    arg.add_argument("--time", dest="time_json", default=None)
    arg.add_argument("--device", dest="device", default=None)
    arg.add_argument("--batch", dest="batch_size", default=1, type=int)
    arg.add_argument("--proc", dest="n_proc", default=int(os.getenv("OMP_NUM_THREADS", 1)), type=int)
    arg.add_argument("--keep-chunks", dest="keep_chunks", default=False, action="store_true",
                    help="Keep temporary chunk files (for debugging)")
    arg.add_argument("--step", dest="step_size", default=1, type=int,
                    help="Step size for trajectory subsampling (default: 1)")
    arg.add_argument("--checkpoint-interval",dest="checkpoint_interval",default=10000,type=int,
                     help="Save checkpoint every N frames and clear temp files (default:10000)")
    arg = arg.parse_args()
    
    timing = {}
    
    # Model setup (same as original)
    timing["loading_ckpt"] = time.time()
    if arg.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arg.device)

    if arg.ckpt_fn is None:
        if arg.cg_model is None:
            raise ValueError("Either --cg or --ckpt argument should be given.")
        else:
            # Model type mapping (same as original)
            if arg.cg_model in ["CalphaBasedModel", "CA", "ca"]:
                model_type = "CalphaBasedModel"
            elif arg.cg_model in ["ResidueBasedModel", "RES", "res"]:
                model_type = "ResidueBasedModel"
            elif arg.cg_model in ["Martini", "martini", "Martini2", "martini2"]:
                model_type = "Martini"
            elif arg.cg_model in ["Martini3", "martini3"]:
                model_type = "Martini3"
            elif arg.cg_model in ["PRIMO", "primo"]:
                model_type = "PRIMO"
            elif arg.cg_model in ["CACM", "cacm", "CalphaCM", "CalphaCMModel"]:
                model_type = "CalphaCMModel"
            elif arg.cg_model in ["CASC", "casc", "CalphaSC", "CalphaSCModel"]:
                model_type = "CalphaSCModel"
            elif arg.cg_model in ["SC", "sc", "sidechain", "SidechainModel"]:
                model_type = "SidechainModel"
            elif arg.cg_model in ["BB", "bb", "backbone", "Backbone", "BackboneModel"]:
                model_type = "BackboneModel"
            elif arg.cg_model in ["MC", "mc", "mainchain", "Mainchain", "MainchainModel"]:
                model_type = "MainchainModel"
            
            if arg.fix_atom:
                arg.ckpt_fn = MODEL_HOME / f"{model_type}-FIX.ckpt"
            else:
                arg.ckpt_fn = MODEL_HOME / f"{model_type}.ckpt"
        
        if not arg.ckpt_fn.exists():
            cg2all.lib.libmodel.download_ckpt_file(model_type, arg.ckpt_fn, fix_atom=arg.fix_atom)
    
    ckpt = torch.load(arg.ckpt_fn, map_location=device)
    config = ckpt["hyper_parameters"]
    timing["loading_ckpt"] = time.time() - timing["loading_ckpt"]
    
    # Model configuration (same as original)
    timing["model_configuration"] = time.time()
    if config["cg_model"] == "CalphaBasedModel":
        cg_model = cg2all.lib.libcg.CalphaBasedModel
    elif config["cg_model"] == "ResidueBasedModel":
        cg_model = cg2all.lib.libcg.ResidueBasedModel
    elif config["cg_model"] == "Martini":
        cg_model = cg2all.lib.libcg.Martini
    elif config["cg_model"] == "Martini3":
        cg_model = cg2all.lib.libcg.Martini3
    elif config["cg_model"] == "PRIMO":
        cg_model = cg2all.lib.libcg.PRIMO
    elif config["cg_model"] == "CalphaCMModel":
        cg_model = cg2all.lib.libcg.CalphaCMModel
    elif config["cg_model"] == "CalphaSCModel":
        cg_model = cg2all.lib.libcg.CalphaSCModel
    elif config["cg_model"] == "SidechainModel":
        cg_model = cg2all.lib.libcg.SidechainModel
    elif config["cg_model"] == "BackboneModel":
        cg_model = cg2all.lib.libcg.BackboneModel
    elif config["cg_model"] == "MainchainModel":
        cg_model = cg2all.lib.libcg.MainchainModel
    
    if arg.is_all and config["cg_model"] in ["PRIMO", "Martini", "Martini3"]:
        topology_map = read_coarse_grained_topology(config["cg_model"].lower())
    else:
        topology_map = None
    
    config = cg2all.lib.libmodel.set_model_config(config, cg_model, flattened=False)
    model = cg2all.lib.libmodel.Model(config, cg_model, compute_loss=False)
    
    state_dict = ckpt["state_dict"]
    for key in list(state_dict):
        state_dict[".".join(key.split(".")[1:])] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.set_constant_tensors(device)
    model.eval()
    timing["model_configuration"] = time.time() - timing["model_configuration"]
    
    # ADD SINGLE PDB PROCESSING HERE (like original cg2all)
    if arg.in_dcd_fn is None:
        # Single PDB processing (copied from original cg2all)
        timing["loading_input"] = time.time()
        input_s = PredictionData(
            arg.in_pdb_fn,
            cg_model,
            topology_map=topology_map,
            dcd_fn=arg.in_dcd_fn,
            radius=config.globals.radius,
            chain_break_cutoff=0.1 * arg.chain_break_cutoff,
            is_all=arg.is_all,
            fix_atom=config.globals.fix_atom,
            batch_size=arg.batch_size,
        )
        
        if len(input_s) > 1 and (arg.n_proc > 1 or arg.batch_size > 1):
            input_s = dgl.dataloading.GraphDataLoader(
                input_s, batch_size=arg.batch_size, num_workers=arg.n_proc, shuffle=False
            )
        else:
            input_s = dgl.dataloading.GraphDataLoader(
                input_s, batch_size=1, num_workers=1, shuffle=False
            )
        
        t0 = time.time()
        batch = next(iter(input_s)).to(device)
        timing["loading_input"] = time.time() - t0
        
        t0 = time.time()
        with torch.no_grad():
            R = model.forward(batch)[0]["R"]
        timing["forward_pass"] = time.time() - t0
        
        timing["writing_output"] = time.time()
        traj_s, ssbond_s = create_trajectory_from_batch(batch, R)
        output = patch_termini(traj_s[0])
        if arg.standard_names:
            standardize_atom_name(output)
        output.save(arg.out_fn)
        if arg.outpdb_fn is not None:
            output.save(arg.outpdb_fn)
        if len(ssbond_s[0]) > 0:
            write_SSBOND(arg.out_fn, output.top, ssbond_s[0])
        timing["writing_output"] = time.time() - timing["writing_output"]
        
        # Print timing information
        time_total = sum(timing.values())
        timing["total"] = time_total
        
        print("\nTiming Summary:")
        for step, t in timing.items():
            print(f"  {step}: {t:.2f} seconds")
        
        if arg.time_json is not None:
            with open(arg.time_json, "wt") as fout:
                fout.write(json.dumps(timing, indent=2))
        
        return  # Exit early for single PDB case
    
    # TRAJECTORY PROCESSING (chunked) - rest of original code
    # Create temporary directory for chunks
    temp_dir = tempfile.mkdtemp(prefix="cg2all_chunks_")
    print(f"Using temporary directory: {temp_dir}")
    
    # Extract output basename for chunk naming
    output_basename = os.path.splitext(os.path.basename(arg.out_fn))[0]
    
    try:
        # Split trajectory into chunks
        log_memory("Initial memory") 
        timing["splitting_trajectory"] = time.time()
        chunk_files = split_trajectory_into_chunks(
            arg.in_pdb_fn, arg.in_dcd_fn, arg.chunk_size, temp_dir, output_basename,arg.dcd_last,
        arg.step_size)
        
        timing["splitting_trajectory"] = time.time() - timing["splitting_trajectory"]
        
        # Process each chunk
        timing["forward_pass"] = time.time()
        checkpoint_output_files = []
        reference_pdb_path = None
        processed_chunks = 0 
        current_checkpoint_chunks = [] 
                       # Process remaining chunks normally
        for i, (chunk_pdb_fn, chunk_dcd_fn) in enumerate(chunk_files):
            if i == 0:
                # Process first chunk and create reference PDB
                output_chunk, reference_pdb_path = process_single_chunk(
                    chunk_pdb_fn, chunk_dcd_fn, model, cg_model, config, 
                    topology_map, device, arg.batch_size, arg.n_proc, 
                    arg.chain_break_cutoff, arg.is_all, arg.fix_atom, 
                    arg.standard_names, temp_dir, i, output_basename,output_path=os.path.dirname(arg.out_fn),
                    save_reference_pdb=True
                )
            else:
                # Process remaining chunks normally
                output_chunk = process_single_chunk(
                    chunk_pdb_fn, chunk_dcd_fn, model, cg_model, config, 
                    topology_map, device, arg.batch_size, arg.n_proc, 
                    arg.chain_break_cutoff, arg.is_all, arg.fix_atom, 
                    arg.standard_names, temp_dir, i, output_basename,output_path=os.path.dirname(arg.out_fn),
                    save_reference_pdb=False
                )
                     
            current_checkpoint_chunks.append(output_chunk)
            processed_chunks+=1
                #Check if we need to create a checkpoint
            frames_processed = processed_chunks *arg.chunk_size
            if frames_processed >= arg.checkpoint_interval or i == len(chunk_files)-1:
                         print(f"Creating checkpoint after {frames_processed} frames...")
                         #Create checkpoint filename
                         checkpoint_fn = f"{output_basename}_checkpoint_{len(checkpoint_output_files):04d}.dcd"
                         checkpoint_path = os.path.join(os.path.dirname(arg.out_fn),checkpoint_fn)
                         
                         #Combine current chunks into checkpoint
                         # Get unit cell info from current chunks - but use individual chunk data
                         
                         combine_trajectory_chunks(current_checkpoint_chunks,checkpoint_path) 

                         checkpoint_output_files.append(checkpoint_path)
                         print(f"Checkpoint saved: {checkpoint_path}")
     
                         #Clean up input chunk files too
                         for j in range(max(0, i - len(current_checkpoint_chunks) + 1), i + 1):
                                input_chunk_file = os.path.join(temp_dir, f"{output_basename}_input_chunk_{j:04d}.dcd")
                                if os.path.exists(input_chunk_file):
                                       os.remove(input_chunk_file)
                         print(f"Cleaned up {len(current_checkpoint_chunks)} temporary chunk files")

                         #reset for next checkpoint
                         current_checkpoint_chunks = []
                         processed_chunks = 0
       
        timing["forward_pass"] = time.time() - timing["forward_pass"]
        
        # Combine all checkpoints into final output
        timing["writing_output"] = time.time()
        print("Combining all checkpoints into final trajectory...")
        # Unit cell info will be extracted from individual checkpoints
        log_memory("Before combining checkpoints")
        combine_checkpoint(
            checkpoint_output_files, reference_pdb_path, arg.out_fn
        )
        log_memory("After combining checkpoints")
        # Verify final output was created successfully
        if not os.path.exists(arg.out_fn):
            raise RuntimeError(f"Failed to create final output file: {arg.out_fn}")
        
        print(f"Final trajectory successfully created: {arg.out_fn}")
        
        # Save final frame as PDB if requested
        if arg.outpdb_fn is not None:
            final_traj = mdtraj.load(arg.out_fn,top=reference_pdb_path)
            final_traj[0].save(arg.outpdb_fn)
            print(f"Final PDB saved: {arg.outpdb_fn}")
        
        timing["writing_output"] = time.time() - timing["writing_output"]
        
        # Only clean up checkpoint files AFTER confirming final output exists
        if not arg.keep_chunks:
               print("Final output verified - cleaning up checkpoint files...")
               if os.path.exists(arg.out_fn):
                    for checkpoint_file in checkpoint_output_files:
                          if os.path.exists(checkpoint_file):
                                os.remove(checkpoint_file)
                    print(f"Cleaned up {len(checkpoint_output_files)} checkpoint files")
        
        # Clean up temporary directory only after successful completion
        if not arg.keep_chunks:
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        else:
            print(f"Keeping temporary files in: {temp_dir}")
            
    finally:
        # Only clean up temp directory if something went wrong and it still exists
        if not arg.keep_chunks and os.path.exists(temp_dir):
            print(f"Error occurred - cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
    
    # Print timing information
    time_total = sum(timing.values())
    timing["total"] = time_total
    
    print("\nTiming Summary:")
    for step, t in timing.items():
        print(f"  {step}: {t:.2f} seconds")
    
    if arg.time_json is not None:
        with open(arg.time_json, "wt") as fout:
            fout.write(json.dumps(timing, indent=2))


#memor saving functions
def clear_mdtraj_caches():
    try:
       import mdtraj.core.element as element
       if hasattr(element,'_cache'):
          element._cache.clear()
    except:
          pass


    try: 
       import mdtraj.formats.registry as registry
       if hasattr(registry,'_cache'):
            registry._cache.clear()
    except:
       pass
#memory logger
import subprocess
import time
import psutil
def log_memory(stage):
        timestamp = time.strftime("%H:%M:%S")
        mem = psutil.virtual_memory()
        process = psutil.Process()
        rss_gb = process.memory_info().rss/1024**3
    
        print(f"[{timestamp} {stage:20s} |"
              f"System: {mem.used/1024**3:5.1f}GB used, {mem.cached/1024**3:5.1f}GB cached |"
              f"Process: {rss_gb:5.1f}GB RSS")

#os cache clearing
def drop_file_cache(filename):
   try:
        fd = os.open(filename, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)

        # Linux: use posix_fadvise to tell OS we don't need this file cached
        
        libc = ctypes.CDLL(ctypes.util.find_library('c'))
        POSIX_FADV_DONTNEED = 4

        fd = os.open(filename,os.O_RDONLY)
        libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        os.close(fd)
        
   except Exception as e:
        print(f"Warning: Could not drop cache for {filename}: {e}")

if __name__ == "__main__":
    main()
