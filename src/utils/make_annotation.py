import xml.etree.ElementTree as ET
import csv
import math

def calculate_angle(p1, p2):
    """2点間の傾きを角度（度）で返す（右上がり正）"""
    x1, y1 = map(float, p1.split(','))
    x2, y2 = map(float, p2.split(','))
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(-dy, dx)  # y軸は下向きなので符号反転
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def parse_xml_to_csv(xml_path, csv_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []

    for image in root.findall('image'):
        filename = image.get('name')

        signal_tag = ''
        state_tag = ''
        slope_deg = ''

        # タグ抽出
        for tag in image.findall('tag'):
            label = tag.get('label')
            if label in ['Green', 'Red', 'None']:
                signal_tag = label
            elif label in ['Normal', 'Occlusion', 'Faded', 'Soiled']:
                state_tag = label

        # 傾き計算
        for polyline in image.findall('polyline'):
            if polyline.get('label') == 'WhiteLine':
                points_str = polyline.get('points')
                points = points_str.strip().split(';')
                if len(points) >= 2:
                    slope_deg = round(calculate_angle(points[0], points[-1]), 2)
                break  

        rows.append([filename, signal_tag, state_tag, slope_deg])

    # CSV出力
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'signal', 'state', 'slope_deg'])
        writer.writerows(rows)

xml_file = 'annotations.xml'
csv_file = 'annotation_summary.csv'
parse_xml_to_csv(xml_file, csv_file)
