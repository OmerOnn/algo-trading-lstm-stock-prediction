import json
import random
import shutil
import time
import os, bz2, pickle
from typing import Tuple, Optional, List

def get_random_order_subfolders(folder_path):
    """
    Get all subfolders within a specified folder and return their full paths in a random order.

    Parameters:
    folder_path (str): The path of the folder to search for subfolders.

    Returns:
    list: A list of full paths to each subfolder in random order.
    """
    subfolders = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            subfolders.append(os.path.join(root, dir_name))
    
    random.shuffle(subfolders)
    return subfolders

def extract_config_init(config):
    output_dir = config['output_dir']
    time_dir = config['time_dir']
    number_of_runs = config['number_of_runs']
    interval_size_min = config['interval_size_min']
    interval_size_sec = config['interval_size_sec']
    interval_size_ms = config['interval_size_ms']
    return output_dir, time_dir, number_of_runs, interval_size_min, interval_size_sec, interval_size_ms


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def get_random_order_subfolders_limit(folder_path, number_of_runs, interval_size_min, interval_size_sec, interval_size_ms=0):
    limit_amount = number_of_runs * ((interval_size_min * 60 * 1000) + (interval_size_sec * 1000) + interval_size_ms)
    
    subfolders = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            if int(dir_name) < limit_amount:
                subfolders.append(os.path.join(root, dir_name))
    
    random.shuffle(subfolders)
    return subfolders

def count_direct_subfolders(dir_path):
    if not os.path.isdir(dir_path):
        raise ValueError(f"The path {dir_path} is not a valid directory.")
    subfolders = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    return len(subfolders)

def print_lines(num_lines):
    for _ in range(num_lines):
        print("-----------------------------------------------------------------------",flush=True)

def create_next_interval_folder(base_path, min_interval, sec_interval, ms_interval=0):
    def parse_folder_name(folder_name):
        """Parse folder name to get the interval in milliseconds as an integer."""
        try:
            return int(folder_name)
        except ValueError:
            return None

    def calculate_step(min_interval, sec_interval, ms_interval):
        """Calculate the step size in milliseconds."""
        return (min_interval * 60 * 1000) + (sec_interval * 1000) + ms_interval

    def convert_ms_to_min_sec_ms(milliseconds):
        """Convert milliseconds to minutes, seconds, and milliseconds."""
        minutes = milliseconds // (60 * 1000)
        remaining_ms = milliseconds % (60 * 1000)
        seconds = remaining_ms // 1000
        millis = remaining_ms % 1000
        return minutes, seconds, millis

    # Calculate the step size in ms
    step_size = calculate_step(min_interval, sec_interval, ms_interval)

    # Get list of all folders in the base path
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    # Parse folders and find the latest interval in ms
    latest_interval = -1
    for folder in folders:
        interval = parse_folder_name(folder)
        if interval is not None and interval > latest_interval:
            latest_interval = interval

    # Calculate the next interval in ms
    next_interval = latest_interval + step_size if latest_interval != -1 else 0
    next_folder_name = str(next_interval)
    next_folder_path = os.path.join(base_path, next_folder_name)

    # Create the next folder
    os.makedirs(next_folder_path, exist_ok=True)
    
    # Convert the next interval to minutes, seconds, and milliseconds
    minutes, seconds, millis = convert_ms_to_min_sec_ms(next_interval)

    return next_folder_path, minutes, seconds, millis

def print_total_run_time_minutes(start_time):
    run_time_seconds = time.time() - start_time
    run_time_minutes = run_time_seconds / 60
    return run_time_minutes

def save_dict_run(folder_path, config_dict, file_name):
    init_pkl_file_path = os.path.join(folder_path, f'{file_name}.pkl') 
    with open(init_pkl_file_path, "wb") as pkl_file:
        pickle.dump(config_dict, pkl_file)
        
def create_or_clear_log_file(relative_log_file="../main_logs/error.txt"):
    # Get the absolute path of the log file relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, relative_log_file)
    error_folder = os.path.join(script_dir, "../main_logs/error_folder")
    
    # Ensure the error folder exists
    os.makedirs(error_folder, exist_ok=True)
    
    # Check if the log file exists and is not empty
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        # Determine the latest file number in the error folder
        existing_files = [f for f in os.listdir(error_folder) if f.startswith("error_") and f.endswith(".txt")]
        if existing_files:
            latest_file_num = max(int(f.split('_')[1].split('.')[0]) for f in existing_files)
        else:
            latest_file_num = 0
        new_file_num = latest_file_num + 1
        new_file_name = f"error_{new_file_num}.txt"
        new_file_path = os.path.join(error_folder, new_file_name)
        
        # Move the log file to the error folder with the new file name
        shutil.move(log_file, new_file_path)
    
    # Ensure the log file is cleared by creating an empty file
    with open(log_file, 'w') as file:
        pass

# ---------- Epoch resume & state helpers (NEW) ----------
# We already have: import os, bz2, pickle, time, shutil, json at top of file.

def list_epoch_dirs(root: str) -> List[str]:
    """
    Return epoch subfolders (full paths), sorted by mtime ascending.
    Only looks at epoch dirs that are direct children of 'root' (e.g., '0', '1000', ...).
    """
    if not os.path.isdir(root):
        return []
    dirs = [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]
    dirs = [p for p in dirs if p.split(os.sep)[-1].isdigit()]
    dirs.sort(key=lambda p: os.path.getmtime(p))
    return dirs

def success_marker_path(epoch_dir: str) -> str:
    return os.path.join(epoch_dir, "_SUCCESS")

def is_epoch_complete(epoch_dir: str) -> bool:
    """An epoch is complete iff a success marker exists."""
    return os.path.isfile(success_marker_path(epoch_dir))

def count_completed_epochs(root: str) -> int:
    """Count only epochs that have a _SUCCESS marker."""
    return sum(1 for d in list_epoch_dirs(root) if is_epoch_complete(d))

def count_completed_epochs_up_to(root: str, max_epoch_dir: Optional[str]) -> int:
    """Count completed epochs up to and including max_epoch_dir."""
    if max_epoch_dir is None:
        return 0
    max_epoch_value = int(os.path.basename(os.path.normpath(max_epoch_dir)))
    return sum(
        1
        for d in list_epoch_dirs(root)
        if is_epoch_complete(d) and int(os.path.basename(os.path.normpath(d))) <= max_epoch_value
    )

def get_last_epoch_dir(root: str) -> Optional[str]:
    """Return most recent epoch dir (by mtime) or None if none exist."""
    dirs = list_epoch_dirs(root)
    return dirs[-1] if dirs else None

def state_paths(epoch_dir: str) -> Tuple[str, str]:
    """
    Return (sat_traffic_state, illumination_plan_path)
    """
    return (
        os.path.join(epoch_dir, "sat_traffic_state.pkl.bz2"),
        os.path.join(epoch_dir, "illumination_plan.pkl.bz2"),
    )

def save_epoch_state(
    epoch_dir: str,
    cell_traffic_state,
    illumination_plan,
    save_illumination_plan: bool = True,
) -> None:
    """
    Persist both state objects in the epoch folder (bz2-pickled).
    """
    os.makedirs(epoch_dir, exist_ok=True)
    cpath, ipath = state_paths(epoch_dir)
    with bz2.BZ2File(cpath, "wb") as f:
        pickle.dump(cell_traffic_state, f, protocol=pickle.HIGHEST_PROTOCOL)
    if save_illumination_plan:
        with bz2.BZ2File(ipath, "wb") as f:
            pickle.dump(illumination_plan, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_beamhopping_dataset_snapshot(
    epoch_dir: str,
    illumination_plan,
    cell_backlog_bits,
    file_name: str = "BeamHopping.pkl.bz2",
) -> None:
    """
    Persist the compact BeamHopping data needed by DatasetBuilder.

    The first tuple item used to be the full SatTrafficQueue. It is intentionally
    an empty placeholder here because the dataset uses the compact per-cell
    backlog map and the resume path uses sat_traffic_state.pkl.bz2.
    """
    os.makedirs(epoch_dir, exist_ok=True)
    path = os.path.join(epoch_dir, file_name)
    compact_payload = ({}, illumination_plan or {}, cell_backlog_bits or {})
    with bz2.BZ2File(path, "wb") as f:
        pickle.dump(compact_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

def try_load_state_from_dir(epoch_dir: str):
    """
    Try loading both state files from a given epoch dir.
    Returns (sat_state, illum) or (None, None) if any missing.
    """
    cpath, ipath = state_paths(epoch_dir)
    if os.path.isfile(cpath):
        with bz2.BZ2File(cpath, "rb") as f:
            sat_state = pickle.load(f)
        illum = {}
        if os.path.isfile(ipath):
            with bz2.BZ2File(ipath, "rb") as f:
                illum = pickle.load(f)
        return sat_state, illum
    return None, None

def load_last_state(root: str):
    """
    Load state from the most recent epoch directory that has state files.
    Preference:
      1) Latest directory that is COMPLETE (_SUCCESS exists)
      2) Otherwise, latest directory that contains state files (resume a crash)
      3) Otherwise, return ({}, {})
    """
    dirs = list_epoch_dirs(root)

    # Pass 1: prefer completed epochs
    for d in reversed(dirs):
        if is_epoch_complete(d):
            cs, ip = try_load_state_from_dir(d)
            if cs is not None and ip is not None:
                return cs, ip

    # Pass 2: allow latest incomplete with state (resume crashed epoch)
    for d in reversed(dirs):
        cs, ip = try_load_state_from_dir(d)
        if cs is not None and ip is not None:
            return cs, ip

    # None found
    return {}, {}

def find_last_state_epoch_dir(root: str) -> Optional[str]:
    """
    Return the most recent epoch directory that contains saved state.
    Preference:
      1) latest COMPLETE epoch with state files
      2) latest epoch with state files
      3) None
    """
    dirs = list_epoch_dirs(root)
    for d in reversed(dirs):
        if is_epoch_complete(d):
            cs, _ = try_load_state_from_dir(d)
            if cs is not None:
                return d
    for d in reversed(dirs):
        cs, _ = try_load_state_from_dir(d)
        if cs is not None:
            return d
    return None

def archive_epoch_dirs_after(
    root: str,
    anchor_epoch_dir: Optional[str],
    archive_all_if_no_anchor: bool = False,
) -> List[str]:
    """
    Move numeric epoch directories after the anchor into a side archive folder.
    This preserves old outputs while letting resume continue from the anchor.
    """
    if anchor_epoch_dir is None:
        if not archive_all_if_no_anchor:
            return []
        anchor_value = -1
    else:
        anchor_value = int(os.path.basename(os.path.normpath(anchor_epoch_dir)))
    archived: List[str] = []
    archive_root = os.path.join(root, "_stale_after_resume")
    os.makedirs(archive_root, exist_ok=True)

    for epoch_dir in list_epoch_dirs(root):
        epoch_name = os.path.basename(os.path.normpath(epoch_dir))
        if int(epoch_name) <= anchor_value:
            continue
        target = os.path.join(archive_root, epoch_name)
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.move(epoch_dir, target)
        archived.append(target)
    return archived

def write_success_marker(epoch_dir: str) -> None:
    """
    Create a _SUCCESS file to mark an epoch complete.
    Call this only AFTER saving the state and any other outputs you require.
    """
    with open(success_marker_path(epoch_dir), "w") as f:
        f.write("ok\n")
