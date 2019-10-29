"""
Organising the results of GEETUP project.
"""

import glob
import os

from kernelphysiology.utils import path_utils
from kernelphysiology.dl.pytorch.geetup import geetup_db
from kernelphysiology.dl.geetup.geetup_utils import map_point_to_image_size


def gather_all_parts_dir(exp_type='validation', **kwargs):
    in_dir = _get_network_type_dir(
        kwargs['db_type'], kwargs['results_dir'], kwargs['network_type']
    )
    all_networks = sorted(glob.glob(in_dir + '/*/'))
    for net_name in all_networks:
        net_name = path_utils.get_folder_name(net_name)
        kwargs['net_name'] = net_name
        gather_all_parts(exp_type, **kwargs)


def gather_all_parts(exp_type='validation', **kwargs):
    out_file = _get_out_file_name(
        exp_type, kwargs['db_type'], kwargs['results_dir'],
        kwargs['network_type'], kwargs['net_name']
    )
    if not os.path.isfile(out_file):
        all_results = {}
        for part_num in range(1, 45):
            part_name = 'Part%.3d' % part_num
            all_results[part_name] = match_results_to_input(
                part_num, exp_type, **kwargs,
            )

        path_utils.write_pickle(out_file, all_results)


def _get_network_type_dir(db_type, results_dir, network_type):
    network_type_dir = '%s/%s/%s/sgd/scratch/' % (
        results_dir, db_type, network_type
    )
    return network_type_dir


def _get_out_file_name(exp_type, db_type, results_dir, network_type, net_name):
    out_file = '%s/%s/%s/sgd/scratch/%s/all_%s.pickle' % (
        results_dir, db_type, network_type, net_name, exp_type
    )
    return out_file


def _get_result_file(part_num, exp_type, db_type, results_dir, network_type,
                     net_name):
    result_file = '%s/%s/%s/sgd/scratch/%s/preds_part%.3d_%s.pickle' % (
        results_dir, db_type, network_type, net_name, part_num, exp_type
    )
    return result_file


def match_results_to_input(part_num, exp_type, dataset_dir, db_type,
                           results_dir, network_type, net_name,
                           model_in_size=(180, 320)):
    part_name = 'Part%.3d' % part_num
    db_dile = '%s/%s/%s/validation.pickle' % (dataset_dir, db_type, part_name)
    geetup_info = geetup_db.GeetupDatasetInformative(db_dile)

    result_file = _get_result_file(
        part_num, exp_type, db_type, results_dir, network_type, net_name
    )
    model_preds = path_utils.read_pickle(result_file)

    current_part_res = {'1': dict(), '2': dict()}
    for j in range(geetup_info.__len__()):
        f_path, f_gt = geetup_info.__getitem__(j)
        f_path = f_path[-1]
        f_gt = f_gt[-1]
        splitted_parts = f_path.replace('//', '/').split('/')
        folder_name = splitted_parts[-2]
        image_name = splitted_parts[-1]
        if '/segments/1/' in f_path:
            seg = '1'
        elif '/segments/2/' in f_path:
            seg = '2'
        else:
            exit('Ups unrecognised segment')
        pred = model_preds[j]
        pred = map_point_to_image_size(pred, (360, 640), model_in_size)
        if folder_name not in current_part_res[seg]:
            current_part_res[seg][folder_name] = []
        sum_error = (f_gt[0] - pred[0]) ** 2 + (f_gt[1] - pred[1]) ** 2
        euc_error = float(sum_error) ** 0.5
        current_part_res[seg][folder_name].append(
            [image_name, f_gt, pred, euc_error]
        )
    return current_part_res
