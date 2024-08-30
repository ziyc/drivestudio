# Modified from: https://github.com/pjlab-ADG/nr3d_lib/
import os
import shutil
import logging

logger = logging.getLogger()

def backup_folder(
    backup_dir: str,
    source_dir: str="./",
    filetypes_to_copy=[".py", ".h", ".cpp", ".cuh", ".cu", ".sh"]
):
    filetypes_to_copy = tuple(filetypes_to_copy)
    os.makedirs(backup_dir, exist_ok=True)
    for file in os.listdir(source_dir):
        if not file.endswith(filetypes_to_copy):
            continue
        source_file_path = os.path.join(source_dir, file)
        target_file_path = os.path.join(backup_dir, file)
        shutil.copy(source_file_path, target_file_path)

def backup_folder_recursive(
    backup_dir: str,
    source_dir: str="./",
    filetypes_to_copy=[".py", ".h", ".cpp", ".cuh", ".cu", ".sh"]
):
    filetypes_to_copy = tuple(filetypes_to_copy)
    for root, _, files in os.walk(source_dir):
        for file in files:
            if not file.endswith(filetypes_to_copy):
                continue
            source_file_path = os.path.join(root, file)
            # Keeps original directory structure
            target_file_path = os.path.join(backup_dir, os.path.relpath(source_file_path, source_dir))
            target_dir_path = os.path.dirname(target_file_path)
            os.makedirs(target_dir_path, exist_ok=True)
            shutil.copy(source_file_path, target_file_path)

def backup_project(
    backup_dir: str,
    source_dir: str="./",
    subdirs_to_copy=["app", "code_multi", "code_single", "dataio", "nr3d_lib"], 
    filetypes_to_copy=[".py", ".h", ".cpp", ".cuh", ".cu", ".sh"],
):
    filetypes_to_copy = tuple(filetypes_to_copy)
    # Automatic backup codes
    logger.info(f"=> Backing up from {source_dir} to {backup_dir}...")
    # Backup project root dir, depth = 1
    backup_folder(backup_dir, source_dir, filetypes_to_copy)
    # Backup cared subdirs, depth = inf
    for subdir in subdirs_to_copy:
        sub_source_dir = os.path.join(source_dir, subdir)
        sub_backup_dir = os.path.join(backup_dir, subdir)
        backup_folder_recursive(sub_backup_dir, sub_source_dir, filetypes_to_copy)
    logger.info("done.")