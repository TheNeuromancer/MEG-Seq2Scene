def num2sub_name(num, all_subjects):
  if len(num) > 2:
    return num
  else: 
    sub_name = [sub for sub in all_subjects if num == sub[0:2]]
    assert len(sub_name) == 1
    return sub_name[0]

tmin_tmax_dict = {"localizer": [-.5, 1.], "one_object": [-.5, 4.], "two_objects": [-.5, 8.]}


TRIG_DICT = {"localizer_block_start": 70,
             "one_object_block_start": 100,
             "two_objects_block_start": 200, 
             "localizer_trial_start": 75,
             "one_object_trial_start": 105,
             "two_objects_trial_start": 205,
             "localizer_pause_start": 80,
             "one_object_pause_start": 110,
             "two_objects_pause_start": 210,
             "new_word": 10, "image": 20, 
             "correct": 30, "wrong": 40}


ica_eog = {"theo_one_object": [0,1,2], 
           "theo_localizer": [0],
           "theo_imgloc": [0],
           "theo_two_objects": [0,1]}