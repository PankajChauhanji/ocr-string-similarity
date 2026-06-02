# This file contains code for calculating similarity between two images.
# This code calculate difference in two images in terms of distance.
# Example posting a local image file:

# import requests
import  time
import imagehash
from PIL import Image
from collections import Counter
import numpy as np

# Using Histogram.
def get_hist_image(path):
    im1 = Image.open(path)
    array1 = np.array(im1)
    flat_array_1 = array1.flatten()
    RH1 = Counter(flat_array_1)
    
    H1 = []
    for i in range(256):
        if i in RH1.keys():
            H1.append(RH1[i])
        else:
            H1.append(0)
    
    return H1

def L2Norm(H1,H2):
    distance =0
    for i in range(len(H1)):
        distance += np.square(H1[i]-H2[i])
    return np.sqrt(distance)

def distance_eucledian_hist(path1, path2):
    # We are calculating image differences using Histogram and Eucledian Distance.
    try:
        H1 = get_hist_image(path1)
        H2 = get_hist_image(path2)
        distance = round(L2Norm(H1, H2)/10000, 2)
        return distance

    except Exception as e:
        print("Error: ", e)
   

# Image Hashing.

def distance_image_hash(path1, path2):
    try:
        ahash1 = imagehash.average_hash(Image.open(path1))
        ahash2 = imagehash.average_hash(Image.open(path2))

        # We can also use phash and whash.
        # But as per out test, ahash works better.

        # phash1 = imagehash.phash(Image.open(path1))
        # phash2 = imagehash.phash(Image.open(path2))
        # whash1 = imagehash.whash(Image.open(path1))
        # whash2 = imagehash.whash(Image.open(path2))
        
        score = ahash1-ahash2
        return score
    except Exception as e:
        print("Error: ", e)
        return -1

def get_image_score(letters, algo="image_hash"):
    """
    This function provide score for similarity between two images.
    Input: (letters = Characters -> type(str), algo = Algorithm name to use"
            Algorithms options: [image_hash, euclid_histogram]
    Output: A score which show similarity between two images and 
            maximum distance found between all images.
            score--->> [0, 1] If score is 0 then they are very similar
                        else if score is towards 1 then they are very different.
    """
    done = []
    scores = {}
    max = 1
    
    for char1 in letters:
        temp = {}
        for char2 in letters:
            if char1 == char2 or char1+char2 in done or char2+char1 in done:
                continue
            else:
                path1 = "images/image_" + char1 + "_.png"
                path2 = "images/image_" + char2 + "_.png"

                distance = -1
                if algo=="image_hash":
                    distance = distance_image_hash(path1, path2)
                elif algo == "euclid_histogram":
                    distance = distance_eucledian_hist(path1, path2)
                else:
                    print("Algo Cannot be detected")
                    continue
                
                if distance and distance!=-1:
                    temp[char2] = distance
                    done.append(char1+char2)
                
                    if max<distance:
                        max = distance
        scores[char1] = temp
                    
    return scores, max

############################################################################################
    
if __name__ == '__main__':
    print("start: ")
    # Generate Similarity Between Different Characters"
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz!@#$%&(){[}]|?><"
    
    # Using Image hashing.
    score, max = get_image_score(letters, "image_hash")
    print(max, "\n", score)

    # Result:
    # After observing all results average hashing gives best result.
    
            
        
    
    