from ultralytics import SAM

# Load a model
model = SAM("/home/qzq/下载/sam2.1_l.pt")

# Display model information (optional)
model.info()

# Run inference with bboxes prompt
results = model("path/to/image.jpg", bboxes=[100, 100, 200, 200])

# Run inference with single point
results = model(points=[900, 370], labels=[1])

# Run inference with multiple points
results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

# Run inference with multiple points prompt per object
results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

# Run inference with negative points prompt
results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])