from utils import generateResultsBulk, generateResultsSingleImage
import sys



def main():
    if(len(sys.argv) !=4):
        print("Not enough arguments. example 'python3 main.py {path to dry image} {path to exposed image} {path to store the result}'")
    # dataset = sys.argv[1]
    # results = sys.argv[2]
    base_image_path  = sys.argv[1]
    image_path  = sys.argv[2]
    result_filepath  = sys.argv[3]
    try:
        # generateResultsBulk(dataset, results)
        generateResultsSingleImage(base_image_path, image_path, result_filepath)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()


