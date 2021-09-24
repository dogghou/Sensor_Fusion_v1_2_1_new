import xml.etree.ElementTree as ET
def load_config(location,ip=''):
    result = {}
    track_info = {}
    filename = '/home/user/Sensor_Fusion/config/{}.xml'.format(location)
    tree = ET.parse(filename)
    root = tree.getroot()
    result['Location'] = root.get('name')
    if ip:
       result['ip'] = ip
    for node in root:
        if node.tag == 'Map_area':
           result['X_lim'] = eval(node.find('X_lim').text)
           result['Y_lim'] = eval(node.find('Y_lim').text)
        elif node.tag == 'Sensor':
           if node.get('ip') == ip:
              track = node.find('track')
              for cls in track:
                  num = cls.get('number')
                  type = cls.find('type').text
                  init_life = cls.find('init_life').text
                  full_blood = cls.find('full_blood').text
                  threshold = cls.find('threshold').text
                  track_info[num] = {'type': type, 'init_life': int(init_life),'full_blood': int(full_blood),'threshold': threshold}
              if node.get('type') == 'Radar':
                 result['range'] = eval(node.find('range').text)
                 result['space'] = eval(node.find('space').text)
              if node.get('type') == 'Camera':
                 detect_area =node.find('detect_area')
                 result['x_offset'] = detect_area.find('x_offset').text
                 result['y_offset'] = detect_area.find('y_offset').text
              result['track'] = track_info
    return result

# print(load_config('tuanjielu'))
# print(load_config('tuanjielu','10.130.210.9'))