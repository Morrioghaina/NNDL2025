import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

class DataLoader {
    constructor() {
        this.images = [];
        this.labels = [];
        this.labelMap = new Map();
        this.reverseLabelMap = new Map();
        this.imageSize = 128;
    }

    async loadDataset(zipFile, csvFile) {
        try {
            // Parse CSV file first to get label mapping
            await this.parseCSV(csvFile);
            
            // Extract and process images from ZIP
            await this.extractImages(zipFile);
            
            // Preprocess data
            return this.preprocessData();
        } catch (error) {
            console.error('Error loading dataset:', error);
            throw error;
        }
    }

    async parseCSV(csvFile) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csvText = e.target.result;
                    const lines = csvText.split('\n').filter(line => line.trim());
                    
                    // Skip header if exists
                    const startIndex = lines[0].includes('image') || lines[0].includes('label') ? 1 : 0;
                    
                    // Create label mapping from all labels in CSV
                    const uniqueLabels = [...new Set(lines.slice(startIndex).map(line => {
                        const parts = line.split(',');
                        return parts[1] ? parts[1].trim() : '';
                    }))].filter(label => label);
                    
                    uniqueLabels.forEach((label, index) => {
                        this.labelMap.set(label, index);
                        this.reverseLabelMap.set(index, label);
                    });
                    
                    // Store filename to label mapping
                    this.fileLabelMap = new Map();
                    lines.slice(startIndex).forEach(line => {
                        const parts = line.split(',');
                        if (parts.length >= 2) {
                            const filename = parts[0].trim();
                            const label = parts[1].trim();
                            if (filename && label) {
                                this.fileLabelMap.set(filename, label);
                            }
                        }
                    });
                    
                    console.log(`Found ${this.fileLabelMap.size} labeled images`);
                    console.log(`Disease classes: ${Array.from(this.labelMap.keys())}`);
                    resolve();
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = reject;
            reader.readAsText(csvFile);
        });
    }

    async extractImages(zipFile) {
        if (typeof JSZip === 'undefined') {
            throw new Error('JSZip library not loaded. Please include JSZip in your HTML.');
        }

        const zip = new JSZip();
        const zipContent = await zip.loadAsync(zipFile);
        
        this.images = [];
        this.labels = [];
        this.originalImages = [];
        
        // Process each file in ZIP
        const imageFiles = Object.keys(zipContent.files).filter(name => 
            name.match(/\.(jpg|jpeg|png)$/i) && !zipContent.files[name].dir
        );

        console.log(`Found ${imageFiles.length} images in ZIP`);

        let processedCount = 0;
        for (const filename of imageFiles) {
            const label = this.fileLabelMap.get(filename);
            if (label === undefined) {
                console.warn(`No label found for image: ${filename}`);
                continue;
            }

           
