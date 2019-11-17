from easyfacenet.simple import facenet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
images = ['images/adi.jpeg', 'images/state.jpeg', 'images/state2.jpeg', 'images/erik.jpeg', 'images/state3.jpeg']
aligned = facenet.align_face(images)
comparisons = facenet.compare(aligned)

print("Is image ADI and ERIK similar? ", bool(comparisons[0][3]))
print("Is image STATE and ERIK similar? ", bool(comparisons[1][3]))
print("Is image STATE and STATE2 similar? ", bool(comparisons[1][2]))
print("Is image STATE and STATE3 similar? ", bool(comparisons[1][4]))