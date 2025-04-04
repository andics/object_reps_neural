import os
import shutil

def copy_selected_subfolders(src_dir, suffix='_copy', allowed_subfolders=None):
    if allowed_subfolders is None:
        allowed_subfolders = {'frames_masks', 'frames_masks_nonmem'}

    # Determine destination directory
    parent_dir, src_name = os.path.split(src_dir.rstrip('/'))
    dst_dir = os.path.join(parent_dir, src_name + suffix)

    if os.path.exists(dst_dir):
        print(f"Destination directory '{dst_dir}' already exists.")
        return

    # Create the destination directory
    os.makedirs(dst_dir)

    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        # Calculate relative path from source root
        rel_path = os.path.relpath(root, src_dir)

        # Only copy allowed subfolders at second level
        path_parts = rel_path.split(os.sep)
        if len(path_parts) == 1 and path_parts[0] == '.':
            # First level, copy all directories (but don't copy files here)
            for d in dirs:
                os.makedirs(os.path.join(dst_dir, d), exist_ok=True)
        elif len(path_parts) == 1:
            # Second level: only create allowed subfolders
            dirs[:] = [d for d in dirs if d in allowed_subfolders]
            for d in dirs:
                os.makedirs(os.path.join(dst_dir, rel_path, d), exist_ok=True)
        elif len(path_parts) >= 2:
            # Deeper levels: copy all files and directories under allowed subfolders
            if path_parts[1] in allowed_subfolders:
                dest_subdir = os.path.join(dst_dir, rel_path)
                os.makedirs(dest_subdir, exist_ok=True)
                # Copy files
                for file in files:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(dest_subdir, file)
                    shutil.copy2(src_file, dst_file)

    print(f"Copied selected folders to '{dst_dir}' successfully.")


# Example usage:
if __name__ == "__main__":
    src_directory = '/EXP_2_TTC/generate_detection_videos_and_meshes/videos_processed'  # <-- Replace this path
    copy_selected_subfolders(src_directory)
