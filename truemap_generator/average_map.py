import os
import json

map_dir = "generated_ground_truth_maps"
map_filenames = ['final_demo_lab3.txt', 'LabC_1_Binh.txt']

def calculate_average(maps):
    average_map = {}
    for key in maps[0].keys():
        x_values = [m[key]['x'] for m in maps]
        y_values = [m[key]['y'] for m in maps]
        average_map[key] = {
            'x': sum(x_values) / len(x_values),
            'y': sum(y_values) / len(y_values),
        }
    return average_map

def read_dict(file_name):
    full_path = os.path.join(map_dir, file_name)
    with open(full_path, 'r') as f:
        data = json.load(f)
    return data

maps = []
for fname in map_filenames:
    maps.append(read_dict(fname))

averaged_map = calculate_average(maps)

# Write the averages to another text file
with open(os.path.join(map_dir, 'average.txt'), 'w') as outfile:
    json.dump(averaged_map, outfile, indent=4)

print(averaged_map)
print(f"Averages have been calculated and written to {os.path.join(map_dir, 'average.txt')}")
