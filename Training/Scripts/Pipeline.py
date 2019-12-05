import operator
import sys
sys.path.append(r"../../Data")
from dataloader import Data
from DetectionModels import *
from SubModels import *
from Training import *

def main():
    # Alexnet3
    # Threshold
    # All SubModels
    # Max is label
    # Compare with test label, can be 0
    # Load our test data
    
    """ FINAL MODEL PATHS HERE """
    # Detector
    detect_path = r"../Models/detect_alex3, imgs=27k, bs=128, epoch=20, lr=0.0001/19_epoch.pt" 
    model_detect = torch.load(detect_path).cuda()
    # Following are hem type models
    # TODO CONFIRM PATH
    # TODO THIS MODEL NAME IS WEIRD
    subdural_path = r"../Models/yeet/alexSubdural_sig,imgs=32k,bs=512,epochs=30,lr=0.01,d0.4/29_epoch.pt" 
    model_subdural = torch.load(subdural_path).cuda().eval()

    intrav_path = r"../Models/yeet/alexIntrav_sig,imgs=32k,bs=32,epochs=30,lr=0.0001,d0.4/29_epoch.pt" 
    model_intrav = torch.load(intrav_path).cuda().eval()

    subara_path = r"../Models/yeet/alexSubara_sig,imgs=32k,bs=512,epochs=30,lr=0.01,d0.4/29_epoch.pt" 
    model_subara = torch.load(subara_path).cuda().eval()

    intrap_path = r"../Models/yeet/alexIntrap_sig,imgs=32k,bs=32,epochs=30,lr=0.0001,d0.4/29_epoch.pt" 
    model_intrap = torch.load(intrap_path).cuda().eval()
    hem_types = { 
            model_subdural : "subdural", 
            model_intrav : "intraventricular", 
            model_subara : "subarachnoid", 
            model_intrap : "intraparenchymal"
    }
    """ FINAL MODEL PATHS """

    test = [
        "../../Data/Processed/train/epidural",
        "../../Data/Processed/val/intraparenchymal",
        "../../Data/Processed/val/subarachnoid",
        "../../Data/Processed/val/intraventricular",
        "../../Data/Processed/val/subdural",
        "../../Data/Processed/val/nohem",    
    ]
    # Do not replace any label
    test_data = Data(test, maximum_per_folder = 50,  tl_model = "alexnet", in_channels=3)
    # Batch size of 1 to simplify
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
    threshold = 0.5
    # Iterate through test_data
    correct = 0
    total = 0
    for img, label in test_data_loader:
        #To Enable GPU Usage
        img = img.cuda()
        label = label.cuda()
        #############################################
        hem_detected = float(torch.nn.functional.sigmoid(model_detect(img)))
        print("Model:", hem_detected)
        if hem_detected <= threshold:
            predictions = {}
            for model, pred_label in hem_types.items():
                fwd_pass = model(img)
                #print(pred_label, fwd_pass)
                
                predictions[pred_label] = float(fwd_pass)
                #predictions[pred_label] = float(torch.nn.functional.sigmoid(fwd_pass))
            # Get probability of Epidural
            #epidural_prob = 0.0
            #for _, prob in predictions.items():
             #   epidural_prob += prob
            #predictions["epidural"] = 1 - epidural_prob
            # Get maxiumum probability
            print(predictions)
            predicted_label = max(predictions, key=predictions.get)

        else:
            # TODO 
            predicted_label = "nohem"
        
        print("Predicted {0} when it was {1}".format( predicted_label,test_data._label_dict[float(label)]))

        if predicted_label == test_data._label_dict[float(label)]: 
            correct += 1
        total += 1
    
    print("Test Accuracy is : " + str(correct/total))

if __name__ == "__main__":
    main()
