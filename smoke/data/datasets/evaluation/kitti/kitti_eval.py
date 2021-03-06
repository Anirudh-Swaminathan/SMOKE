import os
import csv
import logging
import subprocess

from smoke.utils.miscellaneous import mkdir

ID_TYPE_CONVERSION = {
    0: 'Car',
    1: 'Cyclist',
    2: 'Pedestrian'
}


def kitti_evaluation(
        eval_type,
        dataset,
        predictions,
        output_folder,
):
    logger = logging.getLogger(__name__)
    if "detection" in eval_type:
        logger.info("performing kitti detection evaluation: ")
        do_kitti_detection_evaluation(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger
        )


def do_kitti_detection_evaluation(dataset,
                                  predictions,
                                  output_folder,
                                  logger
                                  ):
    predict_folder = os.path.join(output_folder, 'data')  # only recognize data
    mkdir(predict_folder)

    for image_id, prediction in predictions.items():
        predict_txt = image_id + '.txt'
        predict_txt = os.path.join(predict_folder, predict_txt)

        generate_kitti_3d_detection(prediction, predict_txt)

    logger.info("Evaluate on KITTI dataset")
    output_dir = os.path.abspath(output_folder)
    print("---ANI! output_dir - ", output_dir, "---")
    print("---ANI! os.getcwd() - ", os.getcwd(), "---")
    cur_dir = os.getcwd()
    # ch_dir = os.path.join(cur_dir, "./smoke/data/datasets/evaluation/kitti/kitti_eval")
    os.chdir('./smoke/data/datasets/evaluation/kitti/kitti_eval')
    # os.chdir(ch_dir)
    print("---ANI! output_dir after first change - ", output_dir, "---")
    print("---ANI! os.getcwd() after first change - ", os.getcwd(), "---")
    label_dir = getattr(dataset, 'label_dir')
    logger.info("---ANI! label_dir before manual change is {} ---".format(label_dir))
    # TODO Change name to evaluate_object_offline for 40 points
    # TODO Change name to evaluate_object_3d_offline for 11 points
    executable_name = "evaluate_object_3d_offline"
    if not os.path.isfile(executable_name):
        # subprocess.Popen('g++ -O3 -DNDEBUG -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp', shell=True)
        subprocess.call('g++ -O3 -DNDEBUG -o {} {}.cpp'.format(executable_name, executable_name), shell=True)
        logger.info("Compiling executable for {} for first time!".format(executable_name))
    else:
        logger.info("Compiled executable {} already exists!".format(executable_name))
    logger.info("---ANI! label_dir: {} ---\n---ANI! output_dir: {} ---\n".format(label_dir, output_dir))
    label_dir = os.path.join(cur_dir, 'datasets/kitti/training/label_2')
    label_dir = os.path.abspath(label_dir)
    command = "./{} {} {}".format(executable_name, label_dir, output_dir)
    logger.info("---ANI! command: {} ---".format(command))
    output = subprocess.check_output(command, shell=True, universal_newlines=True).strip()
    logger.info(output)
    ch_dir = os.path.join(cur_dir, "tools")
    # os.chdir('../tools')
    os.chdir(ch_dir)
    print("---ANI! output_dir after second change - ", output_dir, "---")
    print("---ANI! os.getcwd() after second change - ", os.getcwd(), "---")


def generate_kitti_3d_detection(prediction, predict_txt):
    with open(predict_txt, 'w', newline='') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        if len(prediction) == 0:
            w.writerow([])
        else:
            for p in prediction:
                p = p.numpy()
                p = p.round(4)
                type = ID_TYPE_CONVERSION[int(p[0])]
                row = [type, 0, 0] + p[1:].tolist()
                w.writerow(row)

    check_last_line_break(predict_txt)


def check_last_line_break(predict_txt):
    f = open(predict_txt, 'rb+')
    try:
        f.seek(-1, os.SEEK_END)
    except:
        pass
    else:
        if f.__next__() == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()
    f.close()
