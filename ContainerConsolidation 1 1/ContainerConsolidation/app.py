from flask import Flask, request, jsonify, render_template
import time
from flask_wtf import FlaskForm
from matplotlib import container
from wtforms import FileField, SubmitField
import numpy as np
from model import PlacementProcedure, BRKGA
import matplotlib.pyplot as plt
from plot import plot_3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
import os
import sys
import glob
import random
import pandas as pd
from werkzeug.utils import secure_filename
warnings.filterwarnings("ignore") 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/files'
file_pattern2 = os.path.join("static/files", '*.csv')
files_to_delete2 = glob.glob(file_pattern2)
for file_path in files_to_delete2:
    try:
        os.remove(file_path) 
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Failed to delete {file_path}: {e}")

class UploadFileForm(FlaskForm):
    file1 = FileField("Box Dimensions in Inches (.csv)")
    file2 = FileField("Container Dimensions in Inches (.csv)")
    submit = SubmitField("Upload Files")

@app.route('/', methods=['GET', 'POST'])
def input_form():
    form = UploadFileForm()
    if form.validate_on_submit():
        file1 = form.file1.data
        file2 = form.file2.data
        if file1:
            file1.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file1.filename)))
        if file2:
            file2.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file2.filename)))
        return render_template('consolidate.html')
    return render_template('index.html', form=form)


@app.route('/result')
def result_form():
    return render_template('result.html')

@app.route('/info')
def placement_info():
    with open('static/output.txt', 'r') as file:
        output_text = file.read()
    
    output_text = output_text.lstrip()
    return render_template('info.html', output_text=output_text)

@app.route('/plots')
def display_png_files():
    directory = 'static'
    png_files = [file for file in os.listdir(directory) if file.endswith('.png')]
    return render_template('plots.html', png_files=png_files)

@app.route('/solve', methods=['GET', 'POST'])
def solve():
    if request.method == 'GET':
        return "Send a POST request to this endpoint with the problem input in JSON format."

    if request.method == 'POST':
        file_pattern = os.path.join("static", '*.png')
        files_to_delete = glob.glob(file_pattern)
        for file_path in files_to_delete:
            try:
                os.remove(file_path) 
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        output_file = os.path.join("static", 'output.txt')
        sys.stdout = open(output_file, 'w')
        
        start_time = time.time()

        boxes_path = os.path.join(app.config['UPLOAD_FOLDER'], 'boxes.csv')
        containers_path = os.path.join(app.config['UPLOAD_FOLDER'], 'containers.csv')
        
        boxes_df = pd.read_csv(boxes_path)
        containers_df = pd.read_csv(containers_path)
        
        # Extracting data
        p = boxes_df['length'].tolist()
        q = boxes_df['width'].tolist()
        r = boxes_df['height'].tolist()
        stackable = boxes_df['stackable'].tolist()
        weights = boxes_df['weight'].tolist()
        
        L = sorted(containers_df['length'].tolist(), reverse=True)
        W = containers_df['width'].tolist()
        H = sorted(containers_df['height'].tolist(), reverse=True)
        limit = sorted(containers_df['weightlimit'].tolist(), reverse=True)

        stackable_indices = [i for i, val in enumerate(stackable) if val == 0]
        inputs_all = {
            'v': list(zip(p, q, r)),
            'V': list(zip(L, W, H))
        }
        inputs = {
            'v': [(p[i], q[i], r[i], weights[i]) for i in stackable_indices],
            'V': list(zip(L, W, H)),
            'limit': list(zip(limit))
        }
        unstackable_indices = [i for i, val in enumerate(stackable) if val == 1]
        inputs_nostack = {
            'v': [(p[i], q[i], r[i]) for i in unstackable_indices]
        }
        model = BRKGA(inputs, num_generations=100, num_individuals=20, num_elites=4, num_mutants=3, eliteCProb=0.7)
        model.fit(patient=3, verbose=True)
        

        inputs['solution'] = model.solution
        decoder = PlacementProcedure(inputs, model.solution)

        V_tuples = [(x[0], x[1], x[2]) for x in inputs['V']]

        draw(decoder, V_tuples)

        inputs['solution'] = model.solution
        decoder = PlacementProcedure(inputs, model.solution)
        fitness = decoder.evaluate()
        container_nos, least_load = divmod(decoder.evaluate(),1)
        container_nos = int(container_nos)
        #least_load = str(decoder.evaluate())
        #least_load = least_load.split(".")[1]
        #percent = least_load[0:2]
        #deci = least_load[2:4]
        #least_load = int(least_load)/100
        print(f"Used {container_nos} containers to consolidate the stackable boxes")
        #print(f"Least Loaded container has {percent}.{deci}% utilization")
        decoder.total_vol()
        decoder.print_weight_info()
        print()
        print("Orientation Details: ")
        print("1 -> lwh")
        print("2 -> lhw")
        print("3 -> wlh")
        print("4 -> whl")
        print("5 -> hlw")
        print("6 -> hwl")
        for container_key, details in decoder.container_details.items():
            print(f'\nDetails for {container_key}:')
            for i, box in enumerate(details['boxes']):
                orientation = details['orientations'][i]
                oriented_box = decoder.revert_orient(box, orientation)

                matching_index = None
                for idx, (p, q, r) in enumerate(inputs_all['v']):
                    if (p, q, r) == oriented_box:
                        matching_index = idx
                        break

                print(f'Box {matching_index + 1} (Dimensions: {oriented_box}, Orientation: {orientation})')
        decoder.print_placement()
        def save_EMS_with_condition(inputs, solution):
            placement = PlacementProcedure(inputs, solution)
            last_container_index = placement.num_opened_containers - 1
            last_container = placement.Containers[last_container_index]

            filtered_EMSs = []
            for EMS in last_container.get_EMSs():
                if EMS[0][2] == 0 and EMS[0][1] == 0:
                    filtered_EMSs.append(EMS)

            return filtered_EMSs
        if unstackable_indices:
            non_stack_space = save_EMS_with_condition(inputs, model.solution)
            if non_stack_space == []:
                end_of_stack = 0
            else:
                end_of_stack = non_stack_space[0][0][0]
            container_len = L[decoder.num_opened_containers-1]
            if container_len == []:
                container_len = 240

            subtracted_EMSs = []
            for EMS in non_stack_space:
                if EMS[0][2] == 0:
                    subtracted_EMSs.append(tuple(np.subtract(EMS[1], EMS[0])))

            num_containers_used = 0

            if subtracted_EMSs == []:
                num_containers_used += 1
                subtracted_EMSs = [(240, 96, 102)]

            
            boxes = [Box(bx[0], bx[1]) for bx in inputs_nostack['v']]

            container_length = subtracted_EMSs[0][0]
            container_width = subtracted_EMSs[0][1]

            all_positions = []

            while boxes:
                positions, not_placed = place_boxes(container_length, container_width, boxes)
                all_positions.append(positions)
                if not_placed:
                    num_containers_used += 1
                    container_length = 240
                    container_width = 96
                boxes = not_placed
            print()
            print()
            print("Placement of Non-Stackable Boxes:")
            print()
            pos = []
            no_stack_boxes = []
            pos2 = []
            no_stack_boxes2 = []
            for i, positions in enumerate(all_positions):
                if i == 0:
                    print("Positions for Empty Space in last container:")
                    for index, (x, y, bx) in positions.items():
                        box_index = [idx for idx, bx_tuple in enumerate(inputs_all['v']) if bx_tuple[:2] == (bx.length, bx.width)][0]
                        print("Box", box_index+1, inputs_all['v'][box_index], "positioned at [",end_of_stack + x,y, " 0] inches from the inner wall of the container")
                        pos.append((end_of_stack+x, y, 0))
                        no_stack_boxes.append(inputs_all['v'][box_index])
                else:
                    print()
                    print("Positions for Additional Container:", i)
                    for index, (x, y, bx) in positions.items():
                        box_index = [idx for idx, bx_tuple in enumerate(inputs_all['v']) if bx_tuple[:2] == (bx.length, bx.width)][0]
                        print("Box", box_index+1, inputs_all['v'][box_index], "positioned at [",x, y, " 0] inches from the inner wall of the container")
                        pos2.append((x, y, 0))
                        no_stack_boxes2.append(inputs_all['v'][box_index])
            print("Total Containers Used:", len(decoder.container_details.items())+ num_containers_used)

            dict1 = {}
            dict2 = {}

            current_key = 1
            dict1[current_key] = []
            dict2[current_key] = []

            for ps, box in zip(pos2, no_stack_boxes2):
                if ps == (0, 0, 0):
                    current_key += 1
                    dict1[current_key] = []
                    dict2[current_key] = []
                    dict1[current_key].append(ps)
                    dict2[current_key].append(box)
                else:
                    dict1[current_key].append(ps)
                    dict2[current_key].append(box)

            dict1 = {k: v for k, v in dict1.items() if v}
            dict2 = {k: v for k, v in dict2.items() if v}

            dict1 = {i: dict1[k] for i, k in enumerate(sorted(dict1.keys()), start=1)}
            dict2 = {i: dict2[k] for i, k in enumerate(sorted(dict2.keys()), start=1)}
            plot_non_stack(pos, no_stack_boxes, container_len, 0, decoder)
            for i in range(num_containers_used):
                plot_non_stack(dict1[i+1], dict2[i+1], container_len, i+1, decoder)
        response = {
            "message": "Solution computed successfully",
            "time_taken": time.time() - start_time
        }
        sys.stdout.close()
        if response["message"] == "Solution computed successfully":
            return jsonify(response), 200
        else:
            return jsonify({"error": "An error occurred"}), 500

def plot_non_stack(pos, no_stack_boxes, container_len, i, decoder):
    positions = pos
    sizes = no_stack_boxes

    # Generate a random color for each box
    colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in sizes]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    pc = plotCubeAt2(positions, sizes, colors=colors, edgecolor="k")
    ax.add_collection3d(pc)

    legend_handles = []
    for size, color in zip(sizes, colors):
        legend_handles.append(plt.Rectangle((0,0),1,1,color=color, label=f"Box: [{size[0]}, {size[1]}, {size[2]}]"))
    ax.legend(handles=legend_handles ,loc = 'upper right')

    ax.set_xlim(pos[0][0], container_len)
    ax.set_ylim(0, 96)
    ax.set_zlim(0, 102)
    ax.view_init(azim=45)
    x_range = container_len - pos[0][0]
    y_range = 96
    if pos[0][0] == 240:
        z_range = 102
    else:
        z_range = 114
    max_range = max(x_range, y_range, z_range)
    aspect = [x_range/max_range, y_range/max_range, z_range/max_range]

    ax.set_box_aspect(aspect)
    if i == 0:
        title = f"Empty Space in Container {i + len(decoder.container_details.items())}"
    else:
        title = f"Container {i + len(decoder.container_details.items())}:"
    ax.set_title(title)
    save_path = os.path.join('static', f"unstackable_{i+1}.png")
    plt.savefig(save_path)
    plt.close(fig)


class Box:
    def __init__(self, length, width):
        self.length = length
        self.width = width

def place_boxes(container_length, container_width, boxes):
    positions = {}
    not_placed = []

    def can_place(box, x, y):
        if x + box.length > container_length or y + box.width > container_width:
            return False
        for pos_x, pos_y, bx in positions.values():
            if (x < pos_x + bx.length and x + box.length > pos_x and
                    y < pos_y + bx.width and y + box.width > pos_y):
                return False
        return True

    def place(box, x, y):
        positions[len(positions) + 1] = (x, y, box)

    for bx in boxes:
        placed = False
        for x in range(container_length):
            for y in range(container_width):
                if can_place(bx, x, y):
                    place(bx, x, y)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            not_placed.append(bx)

    return positions, not_placed
       


def draw(decoder, V_tuples):
    for i in range(decoder.num_opened_containers):
        container = plot_3D(i+1, V=V_tuples[i])
        for box in decoder.Containers[i].load_items:
            container.add_box(box[0], box[1])
        container.show(fig = container.fig, filename = f"container_{i + 1}.png")


def cuboid_data2(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt2(positions, sizes=None, colors=None, **kwargs):
    if not isinstance(colors, (list, np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes, (list, np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data2(p, size=s))
    return Poly3DCollection(np.concatenate(g),facecolors=np.repeat(colors,6), alpha=0.5, **kwargs)

if __name__ == '__main__':
    app.run(debug=True)
