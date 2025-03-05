import xml.etree.ElementTree as ET
import csv

def convert_xml_to_mot(xml_path, output_txt):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    mot_data = []
    
    for track in root.findall(".//track[@label='car']"):
        track_id = int(track.get("id"))
        
        for box in track.findall("box"):
            frame = int(box.get("frame")) + 1  
            xtl = int(float(box.get("xtl")))
            ytl = int(float(box.get("ytl")))
            xbr = int(float(box.get("xbr")))
            ybr = int(float(box.get("ybr")))
            
            width = xbr - xtl
            height = ybr - ytl
            
            conf = 1  
            x, y, z = -1, -1, -1  
            
            mot_data.append([frame, track_id, xtl, ytl, width, height, conf, x, y, z])
    

    with open(output_txt, "w", newline='') as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in sorted(mot_data):  
            writer.writerow(row)


convert_xml_to_mot("ai_challenge_s03_c010-full_annotation.xml", "gt.txt")