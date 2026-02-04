import random

def get_rand_hash(prefix):
    temp_id =  random.getrandbits(128)
    temp_id = "%032x" % temp_id
    return f'{prefix}_{temp_id}'

def generate_full_log_jsonl(data_list,profile_list):

    return_list = []
    for data,profile in zip(data_list,profile_list):
        profile_meta_data= {
            'full_profile': profile,
            'meta_data':{
                    'step_0':
                    {
                        'initial_demographic_seed':data['profile_info_metadata']['initial_demographic_seed'],
                        'event_seed':data['profile_info_metadata']['events_seed']
                    },
                    'step_1':
                    {
                    'generate_profile_string': data['profile_info_metadata']['generated_profile_string']
                    }
            

            }

        }
        return_list.append(profile_meta_data)

    return return_list


def generate_profiles(data_list):
    

    return_list = []
    for data in data_list:
    
        profile_dict = {'profile':data['profile_info_metadata']['profile'],
                        'event_list':[]
                        
                        }

        profile_id = get_rand_hash('profile_id')
        profile_dict['profile']['profile_id'] = profile_id

        profile_dict['profile']['full_name'] = profile_dict['profile']['first_name'] + " " + profile_dict['profile']['last_name']

        events = data['profile_info_metadata']['event_list']
        situations = data['situation_info_metadata_list']
        outputs = data['outputs_list']

        new_events = []

        ##doing events separately to duplicate for implicit event
        for i,event in enumerate(events):
            event_id = get_rand_hash('event_id')
            event['event_id'] = event_id

            if list(event.keys()).count('details') == 1:
                event['text'] = event['details']
                del event['details']
            else:
                event['text'] = 'N/A'

            new_events.append(event)




        for i,(event,situation,output) in enumerate(zip(new_events,situations,outputs)):


            situation_id = get_rand_hash('situation_id')

            situation['text'] = situation['situation']
            situation['situation_id'] = situation_id
            del situation['situation'], situation['attributes'] 

            output_list = []
            
            data_types = output['data_types']
                

            for key in output['output'].keys():
                output_id = get_rand_hash('record_id')
                dt = data_types[key] if data_types else None 
                output_list.append({'record_type': key, 'text': output['output'][key], 'record_id': output_id,'data_type': dt})
            
            
            profile_dict['event_list'].append({'event':event, 'situation':situation, 'record_list':output_list})

        return_list.append(profile_dict)

    return return_list


def generate_unrolled_profiles(data_list):

    return_list = []

    for data in data_list:
        for entry in data['event_list']:
            for output in entry['record_list']:
                return_list.append({'profile':data['profile'], 'event':entry['event'],'situation':entry['situation'],'record':output})

            

    return return_list