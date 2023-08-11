import torch
from ultralytics import YOLO

#checkpoint is the path to my best model from previous training
#checkpoint_path = r"\\VAPP-SRV-FS3\axc22$\Projects\Image_recognition\runs\detect\train\weights\best.pt"
##checkpoint = torch.load(checkpoint_path)

#creates and loads new model to be best prev trained
#model = YOLO("yolov8n.yaml")
#model.load_state_dict(checkpoint['model'])

#continues training
#results = model.train(data="config.yaml", epochs=40)

#save model to project file once done training
#model.save("\\VAPP-SRV-FS3\axc22$\Projects\Image_recognition")
#torch.save(results, "\\VAPP-SRV-FS3\axc22$\Projects\Image_recognition")





#code to print a checkpoints keys, used for loading
#print(checkpoint.keys())



# # Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# # Use the model
results = model.train(data="config.yaml", epochs=35)  # train the model

# #saves the model to the project file
model.save("\\VAPP-SRV-FS3\axc22$\Projects\Image_recognition")

# #saves the results for future use to same location as model
torch.save(results, "\\VAPP-SRV-FS3\axc22$\Projects\Image_recognition")



#if the training doesn't work, delete error files
#can be found at the top of the training in terminal


