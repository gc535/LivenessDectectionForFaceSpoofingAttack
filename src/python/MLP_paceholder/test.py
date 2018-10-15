import os
import numpy as np


def importFromTXT(dirpath, filename):
    assert(phase=="Test" or phase=="Train")

    rootDir = os.path.join(os.getcwd(), "../../..")
    binDir = os.path.join(rootDir, "bin")

    if  os.path.exists(os.path.join( binDir, phase+"Feature.txt")) \
        and os.path.exists(os.path.join( binDir, phase+"Label.txt")) :
        data_x = np.loadtxt(os.path.join( binDir, phase+"Feature.txt"), dtype='f')
        data_y = np.loadtxt(os.path.join( binDir, phase+"Label.txt"), dtype=int).reshape(-1, 1)
        print(len(data_x), len(data_x[0]))
        print(len(data_y), len(data_y[0]))
        return data_x, data_y

    else:
        print("[ERROR]: No datafile found in ./bin. Action aborted.")
        exit()

importFromTXT("Test")

"""
def parseXML(xmlfile): 
  
    # create element tree object 
    tree = ET.parse(xmlfile) 
  
    # get root element 
    root = tree.getroot() 
  
    # create empty list for news items 
    newsitems = [] 
  
    # iterate news items 
    for item in root.findall('./channel/item'): 
  
        # empty news dictionary 
        news = {} 
  
        # iterate child elements of item 
        for child in item: 
  
            # special checking for namespace object content:media 
            if child.tag == '{http://search.yahoo.com/mrss/}content': 
                news['media'] = child.attrib['url'] 
            else: 
                news[child.tag] = child.text.encode('utf8') 
  
        # append news dictionary to news items list 
        newsitems.append(news) 
      
    # return news items list 
    return newsitems
"""