"""

"""

from torch.utils import data as torch_data

class OddOneOutTrain(torch_data.Dataset):

    def __init__(self, root, transform=None):
        self.root = root


    def __getitem__(self, item):
        se_file_path = self.valid_trials[item]

        if self.transform is not None:
            trial_img = self.transform(trial_img)

        # -1 because in Matlab they're stored from 1
        mass_dist = self.stimuli_data[tnum][2] - 1
        intensity = self.stimuli_data[tnum][-3]
        # -1 because in Matlab they're stored as 1 and 2
        response = self.stimuli_data[tnum][-1] - 1

        # converting to tensor
        # FIXME: normalise tirl_img to -1 to 1
        trial_img = torch.tensor(trial_img.transpose((2, 0, 1))).type(torch.FloatTensor)
        intensity = torch.tensor([intensity]).type(torch.FloatTensor)

        # FIXME
        # response -1 means the participant hasn't responded
        # in this case we'll assume that zhe hasn't felt it.
        if response == -1:
            # print('Shouldnt happen', se_file_path)
            response = 1
        return trial_img, intensity, mass_dist, response, tnum

    def __len__(self):
        return self.num_trials