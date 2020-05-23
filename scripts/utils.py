from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

def one_hot_encode_along_channel_axis(sequence, onehot_axis=1):
    to_return = np.zeros((len(sequence),4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=onehot_axis)
    return to_return


def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis==0 or one_hot_axis==1
    if (one_hot_axis==0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis==1):
        assert zeros_array.shape[0] == len(sequence)
    #will mutate zeros_array
    for (i,char) in enumerate(sequence):
        if (char=="A" or char=="a"):
            char_idx = 0
        elif (char=="C" or char=="c"):
            char_idx = 1
        elif (char=="G" or char=="g"):
            char_idx = 2
        elif (char=="T" or char=="t"):
            char_idx = 3
        elif (char=="N" or char=="n"):
            continue #leave that pos as all 0's
        else:
            raise RuntimeError("Unsupported character: "+str(char))
        if (one_hot_axis==0):
            zeros_array[char_idx,i] = 1
        elif (one_hot_axis==1):
            zeros_array[i,char_idx] = 1


def enum(**enums):
    class Enum(object):
        pass
    to_return = Enum
    for key, val in enums.items():
        if hasattr(val, '__call__'):
            setattr(to_return, key, staticmethod(val))
        else:
            setattr(to_return, key, val)
    to_return.vals = [x for x in enums.values()]
    to_return.the_dict = enums

    return to_return


def seq_from_onehot(onehot_data):
    sequences = []
    if len(onehot_data.shape) != 3:
        onehot_data = onehot_data[np.newaxis, :]
    for i in range(onehot_data.shape[0]):
        onehot_seq = onehot_data[i, :, :]
        sequence = ''
        if onehot_seq.shape[0] < onehot_seq.shape[1]:
            onehot_seq = np.swapaxes(onehot_seq, 0, 1)
        for j in range(onehot_seq.shape[0]):
            if(onehot_seq[j, 0]==1):
                sequence = sequence + "A"
            elif (onehot_seq[j, 1]==1):
                sequence = sequence + "C"
            elif (onehot_seq[j, 2]==1):
                sequence = sequence + "G"
            elif (onehot_seq[j, 3]==1):
                sequence = sequence + "T"

        sequences.append(sequence)

    return sequences


def reverse_complement_seq(seq):
    table = str.maketrans("ACTG", "TGAC")
    return seq.translate(table)[::-1]


def reverse_complement_onehot(onehot, window_size):
    dim = onehot.shape
    axis_nt = dim.index(4)
    axis_base = dim.index(window_size)
    onehot_rc = np.flip(onehot, axis=axis_nt)
    onehot_rc = np.flip(onehot_rc, axis=axis_base)
    return onehot_rc


def shift_onehot(onehot_data, shift_amount, pad_value=0.0):
    """Shift a sequence left or right by shift_amount.
        Args:
        seq: a [batch_size, sequence_length, sequence_depth] sequence to shift
        shift_amount: the signed amount to shift (tf.int32 or int)
        pad_value: value to fill the padding (primitive or scalar tf.Tensor)
    """
    flag_swap = False
    if len(onehot_data.shape) != 3:
        onehot_data = onehot_data[np.newaxis, :]
        
    if onehot_data.shape[2] > onehot_data.shape[1]:
        onehot_data = np.swapaxes(onehot_data,1,2)
        flag_swap = True
        
    input_shape = onehot_data.shape

    pad = pad_value * np.ones(onehot_data[:, 0:np.abs(shift_amount), :].shape)

    def _shift_right(_onehot_data):
        sliced_onehot_data = _onehot_data[:, :-shift_amount:, :]
        return np.concatenate((pad, sliced_onehot_data), axis=1)

    def _shift_left(_onehot_data):
        sliced_onehot_data = _onehot_data[:, -shift_amount:, :]
        return np.concatenate((sliced_onehot_data, pad), axis=1)

    if shift_amount > 0:
        output = _shift_right(onehot_data)
    else:
        output = _shift_left(onehot_data)

    output = np.reshape(output, input_shape)
    
    if flag_swap:
        output = np.swapaxes(output,1,2)
        
    return output


def metric_pearson(obs, pred):
    correlations = []
    for i in range(len(pred)):
        correlations.append(np.corrcoef(pred[i, :], obs[i, :])[0, 1])
    return correlations


def metric_mse(obs, pred):
    from sklearn.metrics import mean_squared_error
    mses = []
    for i in range(len(pred)):
        mses.append(mean_squared_error(obs[i, :], pred[i, :]))
    return mses


def metric_r2_score(obs, pred):
    r2 = []
    for i in range(len(pred)):
        ssres = np.sum(np.square(obs[i, :] - pred[i, :]))
        sstot = np.sum(np.square(obs[i, :] - np.mean(obs[i, :])))
        r2.append(1 - ssres / sstot)
    return r2


def compute_loss(obs, pred, combine_weighting):
    correlations = metric_pearson(obs, pred)
    mses = metric_mse(obs, pred)
    metric_loss = 1 - np.stack(correlations) + combine_weighting*np.stack(mses)
    return metric_loss


def minmax_norm(var_in):
    max_val = np.amax(var_in)
    min_val = np.amin(var_in)
    subtracted = var_in - min_val
    var_out = subtracted / (max_val - min_val)
    return var_out


def minmax_scale(pred, labels):
    subtracted = pred - np.min(pred, axis=-1)
    max_pred = np.max(subtracted, axis=-1)
    min_pred = np.min(subtracted, axis=-1)
    max_true = np.max(labels, axis=-1)
    min_true = np.min(labels, axis=-1)
    scaled = subtracted / (max_pred - min_pred) * (max_true - min_true) + min_true
    return scaled


def rounding_generator(data, y, name, batch_size): 
    import copy
    l = len(data)
    num_sample = batch_size - l % batch_size    
    data_out = copy.deepcopy(data)
    data_out = np.concatenate((data_out, data_out[0:num_sample,:,:]), axis=0)
    y_out = copy.deepcopy(y)
    y_out = np.concatenate((y_out, y_out[0:num_sample,:]), axis=0)
    name_out = copy.deepcopy(name)
    name_out = np.concatenate((name_out, name_out[0:num_sample]), axis=0)
    
    return data_out, y_out, name_out


# Matching the two datasets by concatenating the first samples from the same dataset
def upsample_generator(data1, data2): 
    l1 = len(data1)
    l2 = len(data2)
    #extending data 1 to the same size of data 2
    sampleRelation = l2 // l1 #l1 must be bigger
    if l2 % l1 > 0:
        sampleRelation += 1
    index_in = list(range(l1))
    index_out = np.concatenate([index_in] * sampleRelation, axis=0)
    index_out = index_out[:l2]  
      
    return index_out


# Use genome relationship, assuming the order is the corresponding
def interpolating_generator(data1, data2): 
    from scipy.interpolate import interp1d
    index_in = np.linspace(0, len(data1), num=len(data1), endpoint=True)
    f = interp1d(index_in, index_in)
    index_new = np.linspace(0, len(data1), num=len(data2), endpoint=True)
    index_out = f(index_new)
    index_out = np.rint(index_out).astype(int) 
    index_out[index_out>=len(data1)] = len(data1) - 1   
    
    return index_out


def list_seq_to_fasta(index_seq, seq, motif_name, flag_unique, output_directory, single_filter_txt):
    if flag_unique:
        output_filename = motif_name + '_seq_unique' + single_filter_txt + '.fasta'      
        index_seq = list(set(index_seq))        
    else:
        output_filename = motif_name + '_seq' + single_filter_txt + '.fasta'       
    list_seq = np.asarray(seq)[index_seq]   
    print(len(index_seq))
    with open(output_directory + output_filename, 'w') as f:
        for i in range(len(index_seq)):
            f.write('>' + str(i) + '\n')
            f.write(list_seq[i] + '\n')


def list_seqlet_to_fasta(index_seq, index_start, seq, motif_name, output_directory, single_filter_txt):
    output_filename = motif_name + '_seqlet' + single_filter_txt + '.fasta'           
    list_seq = np.asarray(seq)[index_seq]   
    print(len(index_seq))
    with open(output_directory + output_filename, 'w') as f:
        for i in range(len(index_seq)):
            max(0, index_start[i])
            seqlet = list_seq[i][max(0, index_start[i]):min(250, index_start[i]+19)]
            f.write('>' + str(i) + '\n')
            f.write(seqlet + '\n')


def mut_seq(mut_dict, onehot_data, loc_txt): 
    output_onehot_data = copy.deepcopy(onehot_data)
    for k in mut_dict.keys():
        seq = mut_dict[k]['seq']
        if loc_txt == 'mut':
            mut_start = mut_dict[k][loc_txt+'_start']
            mut_end = mut_dict[k][loc_txt+'_end'] 
        else:
            mut_start = mut_dict[k][loc_txt+'_start'][0]
            mut_end = mut_dict[k][loc_txt+'_end'][0]
        if output_onehot_data.shape[-1] > output_onehot_data.shape[-2]:
            output_onehot_data[seq, :, mut_start:mut_end+1] = 0  # Not activation
        else:
            output_onehot_data[seq, mut_start:mut_end+1, :] = 0
    return output_onehot_data


def mut_seq_perbase_pernucleotide(mut_dict, onehot_data, loc_txt): 
    len_mutation = mut_dict[0]['mut_end'] - mut_dict[0]['mut_start'] #19
    output_onehot_data = np.zeros((251, 4, len_mutation, 4))
    for k in mut_dict.keys():
        seq = mut_dict[k]['seq']
        if loc_txt == 'mut':
            mut_start = mut_dict[k][loc_txt+'_start']
            mut_end = mut_dict[k][loc_txt+'_end'] 
        else:
            mut_start = mut_dict[k][loc_txt+'_start'][0]
            mut_end = mut_dict[k][loc_txt+'_end'][0]              
        
        for i in range(mut_start, mut_end):
            for j in range(4):
                tmp = copy.deepcopy(onehot_data[seq, :, :]) # 19 by 4 
                if tmp.shape[-1] == 4:
                    tmp[i, :] = 0
                    tmp[i, j] = 1
                else: # what happens when both=4?
                    tmp[:, i] = 0
                    tmp[j, i] = 1
                    
                output_onehot_data[:,:,i-mut_start,j] = tmp
                    
    return output_onehot_data


def mut_seq_perbase_opposite(mut_dict, onehot_data, loc_txt, flag_order): 
    for k in mut_dict.keys():
        seq = mut_dict[k]['seq']
        if loc_txt == 'ocr':
            mut_start = mut_dict[k]['start']
            mut_end = mut_dict[k]['end']
        elif loc_txt == 'mut':
            mut_start = mut_dict[k][loc_txt+'_start']
            mut_end = mut_dict[k][loc_txt+'_end'] 
        elif loc_txt == 'resp':
            mut_start = mut_dict[k][loc_txt+'_start'][0]
            mut_end = mut_dict[k][loc_txt+'_end'][0]              
        
        output_onehot_data = np.zeros((251, 4, mut_end - mut_start))
        for i in range(mut_start, mut_end):
            tmp = copy.deepcopy(onehot_data[seq, :, :])                     
            if tmp.shape[-1] == 4:
                if flag_order == 'ATGC':
                    tmp = tmp[:, [0, 3, 2, 1]]
                tmp_original = np.copy(tmp[i, :])
                if tmp_original[0] or tmp_original[3]: # AT->C;
                    tmp[i, :] = 0
                    tmp[i, 1] = 1
                elif tmp_original[1] or tmp_original[2]: # CG->A
                    tmp[i, :] = 0
                    tmp[i, 0] = 1
            else: # what happens when both=4?
                if flag_order == 'ATGC':
                    tmp = tmp[[0, 3, 2, 1], :]
                tmp_original = np.copy(tmp[:, i])
                if tmp_original[0] or tmp_original[3]: # AT->C;
                    tmp[:, i] = 0
                    tmp[1, i] = 1
                elif tmp_original[1] or tmp_original[2]: # CG->A
                    tmp[:, i] = 0
                    tmp[0, i] = 1 
            output_onehot_data[:, :, i] = tmp
                    
    return output_onehot_data


def mut_seq_perbase_opposite_hyp(mut_dict, onehot_data, hyp_score, loc_txt, flag_order): 
    for k in mut_dict.keys():
        hyp_score_k = np.stack(hyp_score[str(k)])
        seq = mut_dict[k]['seq']
        if loc_txt == 'ocr':
            mut_start = mut_dict[k]['start']
            mut_end = mut_dict[k]['end']
        elif loc_txt == 'mut':
            mut_start = mut_dict[k][loc_txt+'_start']
            mut_end = mut_dict[k][loc_txt+'_end'] 
        elif loc_txt == 'resp':
            mut_start = mut_dict[k][loc_txt+'_start'][0]
            mut_end = mut_dict[k][loc_txt+'_end'][0]              
        
        output_onehot_data = np.zeros((251, 4, mut_end - mut_start))
        for i in range(mut_start, mut_end):
            tmp = copy.deepcopy(onehot_data[seq, :, :]) 
            tmp_hyp = copy.deepcopy(hyp_score_k[seq, :, :])                    
            if tmp.shape[-1] == 4:
                if flag_order == 'ATGC':
                    tmp = tmp[:, [0, 3, 2, 1]]                
                tmp[i, :] = 0
                tmp[i, np.argmin(tmp_hyp[i, :])] = 1

            else: # what happens when both=4?
                if flag_order == 'ATGC':
                    tmp = tmp[[0, 3, 2, 1], :]
                tmp[:, i] = 0
                tmp[np.argmin(tmp_hyp[i, :]), i] = 1
            output_onehot_data[:, :, i] = tmp
                    
    return output_onehot_data


if __name__ == "__main__":
	print('This is util functions for basic operations.')