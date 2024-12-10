# beginning and end of epochs for each block type
tmin_tmax_dict = {"localizer": [-.5, 1.], "one_object": [-.5, 4.], "two_objects": [-.5, 8.], "response_locked": [-2, 1.], "obj": [-.5, 4.], "scenes": [-.5, 8.], "alltogether": [-.5, 8.]}

# trigger values for each block type
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

# old stuff
ica_eog = {"theo_one_object": [0,1,2], 
           "theo_localizer": [0],
           "theo_imgloc": [0],
           "theo_two_objects": [0,1]}


# subjects with chance level acuracy
bad_subjects = ["03", "09", "10", "12", "14"]
bad_subs_dim = ['29', '26', '25', '23', '20', '16', '15', '10', '04', '02']

colors = ["vert", "bleu", "rouge"]
shapes = ["triangle", "cercle", "carre"]

t_confusion = [0.2, .25, .3, .35, 0.4, .45, .55, 0.6, .7, 0.8]