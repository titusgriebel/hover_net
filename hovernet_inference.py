import subprocess
import os

def run_inference(model_dir, input_dir, output_dir, type_info_path):
    for dataset in ['tnbc']:
        # if os.path.exists(os.path.join('/mnt/lustre-grete/usr/u12649/scratch/models/hovernet/inference/', f'{dataset}')):
        #     continue
        input_path = os.path.join(input_dir, dataset, 'loaded_dataset/complete_dataset/images')
        for model in ['consep', 'cpm17', 'kumar', 'pannuke', 'monusac']:
            if os.path.exists(f'/mnt/lustre-grete/usr/u12649/scratch/models/hovernet/inference/{dataset}/{model}'):
                continue
            output_path = os.path.join(output_dir, dataset, model)
            os.makedirs(output_dir, exist_ok=True)
            if model in ['consep', 'cpm17', 'kumar']:
                model_mode = 'original'
                model_path = os.path.join(model_dir, f'hovernet_original_{model}_notype_tf2pytorch.tar')
                nr_types = 0
                type_info = ''
            else:
                print(f'Running {model} inference right now')
                model_mode = 'fast'
                model_path = os.path.join(model_dir, f'hovernet_fast_{model}_type_tf2pytorch.tar')
                type_info = type_info_path
                if model == 'pannuke':
                    nr_types = 6
                else:
                    nr_types = 5

            args = [
                "--nr_types", f"{nr_types}",
                "--type_info_path", f"{type_info}",
                "--model_path", f"{model_path}",
                "--model_mode", f"{model_mode}",  
                "tile",
                "--input_dir", f"{input_path}",
                "--output_dir", f"{output_path}",
                "--save_raw_map"
            ]

            command = ['python3', '/user/titus.griebel/u12649/hover_net/run_infer.py'] + args

            subprocess.run(command)



run_inference(model_dir='/mnt/lustre-grete/usr/u12649/scratch/models/models/hovernet/checkpoints', input_dir='/mnt/lustre-grete/usr/u12649/scratch/data', output_dir='/mnt/lustre-grete/usr/u12649/scratch/models/hovernet/inference/', type_info_path='/user/titus.griebel/u12649/hover_net/type_info.json')
