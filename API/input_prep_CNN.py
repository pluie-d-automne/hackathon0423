import numpy as np

def input_prep_CNN(item):
    """
    Creates tenzor for CNN model from initial json input from the Garpix dataset.
    Takes dict as an input.
    Returns numpy array as an output.
    """
    def prepare_item(item):
        loading_size = item['data_result']['cargo_space']['loading_size']
        norm_base = max(loading_size['width'], loading_size['height'], loading_size['length'])
        for box in item['data_result']['boxes']:
            box_size = box['size']
            box['size_scale'] = {k: round(v*100/norm_base,2) for k,v in box_size.items()}
            box['volume'] = box['size_scale']['width'] * box['size_scale']['height'] * box['size_scale']['length']
        item['data_result']['boxes'] = sorted(item['data_result']['boxes'], key = lambda x: x['volume'], reverse = True)
        item_new = dict()
        item_new['loading_size'] = {k: round(v*100/norm_base,2) for k,v in loading_size.items()}
        item_new['boxes'] = [{"size": box['size_scale'], "stacking": box['stacking'], "turnover": box['turnover']} for box in item['data_result']['boxes']]
        return item_new
    
    def make_tenzor(item):
        tenzoriezed_item = [[item['loading_size']['width'], item['loading_size']['height'], item['loading_size']['length'], False, False]] + \
            [ [box['size']['width'], box['size']['height'], box['size']['length'], box['stacking'], box['turnover']] for box in item['boxes']]
        add_dims = 2850 - len(tenzoriezed_item) #2850 - max boxes in the set
        tenzoriezed_item = np.concatenate((np.array(tenzoriezed_item), np.zeros((add_dims, 5))))
        return np.array([tenzoriezed_item])
    
    return make_tenzor(prepare_item(item))