# Images Directory

This directory is for storing **PCB test images** to evaluate your trained models.

## ⚠️ IMPORTANT: Add your test images here!

### Supported Formats

- `.jpg`, `.jpeg` - JPEG images (recommended)
- `.png` - PNG images  
- `.bmp` - Bitmap images
- `.tiff`, `.tif` - TIFF images

### Usage with GUI

1. **Place your PCB images** in this directory
2. **Launch the GUI**: `python -m gui_test.app`
3. **Load images** using the GUI interface:
   - **Single image**: Click "Load Image" 
   - **Batch processing**: Click "Load Folder"

### Example Structure

```
images/
├── good_pcbs/
│   ├── pcb_good_001.jpg
│   ├── pcb_good_002.jpg
│   └── pcb_good_003.jpg
├── defective_pcbs/
│   ├── pcb_defect_001.jpg
│   ├── pcb_defect_002.jpg
│   └── pcb_defect_003.jpg
├── test_samples/
│   ├── sample_1.jpg
│   └── sample_2.jpg
└── README.md
```

### Getting Test Images

You can get PCB test images from:

1. **Kaggle Dataset**: [PCB Defects - Akhatova](https://www.kaggle.com/datasets/akhatova/pcb-defects)
2. **Your own PCB photos**
3. **Online PCB image databases**

### Tips for Best Results

- **Image quality**: Use clear, well-lit images
- **Resolution**: 640x640 pixels or higher recommended
- **Format**: JPEG format works best
- **Lighting**: Avoid shadows and reflections

The GUI will recursively search subdirectories when using batch processing!