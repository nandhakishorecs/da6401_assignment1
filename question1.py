import wandb 
import numpy as np 
import keras 
import matplotlib.pyplot as plt 


if __name__ == '__main__': 
    # ---------------------------- Wandb credentials ----------------------------
    # wandb.init(
    #     # project="v1.0",             # Version number for better tracking
    # )

    # ---------------------------- Question 1 ----------------------------
    [(train_X, train_y), (test_X, test_y)] = keras.datasets.fashion_mnist.load_data()
    n_classes = 10 

    class_map = {
        0 : "T-shirt/top", 
        1 : "Trouser", 
        2 : "Pullover",
        3 : "Dress", 
        4 : "Coat", 
        5 : "Sandal", 
        6 : "Shirt",
        7 : "Sneaker", 
        8 : "Bag",
        9 : "Ankle boot"  
    }

    plt.figure(figsize = [12, 6])
    images = []
    classes = []

    for i in range(n_classes): 
        postition = np.argmax(train_y == i)
        image = train_X[postition, :, :]
        plt.subplot(2, 5, i+1)
        plt.imshow(image)
        plt.title(class_map[i])
        images.append(image)
        classes.append(class_map[i])
    
    # For tetsing on local machine 
    # plt.savefig('test_image.png')
    
    # ---------------------------- Wandb credentials ----------------------------
    # wandb.log({
    #     "Question 1": [wandb.Image(img, caption = caption) for img, caption in zip(images, classes)]
    # })
    # wandb.finish()