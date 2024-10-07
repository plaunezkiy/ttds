import xml.etree.ElementTree as ET

collection = "sample.xml"
tree = ET.parse(f"data/collections/{collection}")
print(tree)
