import os

def list_files_alphabetically(folder_path):
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        files.sort()
        return files
    except FileNotFoundError:
        print(f"The folder '{folder_path}' was not found.")
        return []

# Example usage
folder = '/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/detr/EXP_2_TTC/generate_detection_videos_and_meshes/exp2TTC_files'
sorted_files = list_files_alphabetically(folder)
print(sorted_files)
