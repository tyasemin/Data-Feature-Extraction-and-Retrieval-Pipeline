#!/bin/bash
# Run SAM Segmentation + CLIP Feature Extraction + Tag Generation Pipeline

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SAM Segmentation + CLIP Feature Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in the SAM directory
if [ ! -f "docker-compose-pipeline.yml" ]; then
    echo -e "${YELLOW}Error: docker-compose-pipeline.yml not found!${NC}"
    echo "Please run this script from the SAM directory"
    exit 1
fi

# Create output directories
mkdir -p segmented_features segmented_tags segmented_images
echo -e "${GREEN}✓${NC} Created output directories"

# Build the Docker image
echo ""
echo -e "${BLUE}Building Docker image...${NC}"
docker build -t sam_clip:latest -f Dockerfile .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Docker image built successfully"
else
    echo -e "${YELLOW}✗${NC} Docker build failed"
    exit 1
fi

# Run the pipeline
echo ""
echo -e "${BLUE}Running SAM + CLIP pipeline...${NC}"
echo -e "${YELLOW}This will process images with SAM segmentation and CLIP features${NC}"
echo ""

docker-compose -f docker-compose-pipeline.yml up --abort-on-container-exit

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   Pipeline completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Results saved to:"
    echo -e "  Features: ${BLUE}./segmented_features/${NC}"
    echo -e "  Tags: ${BLUE}./segmented_tags/${NC}"
    echo -e "  Segments: ${BLUE}./segmented_images/${NC}"
    echo ""
    
    # Count output files
    num_features=$(ls -1 segmented_features/*.json 2>/dev/null | wc -l)
    num_tags=$(ls -1 segmented_tags/*.json 2>/dev/null | wc -l)
    num_segments=$(ls -1 segmented_images/*.png 2>/dev/null | wc -l)
    
    echo -e "Generated:"
    echo -e "  ${GREEN}${num_features}${NC} feature files"
    echo -e "  ${GREEN}${num_tags}${NC} tag files"
    echo -e "  ${GREEN}${num_segments}${NC} segment images"
else
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}   Pipeline encountered errors${NC}"
    echo -e "${YELLOW}========================================${NC}"
fi

# Cleanup
echo ""
echo -e "${BLUE}Cleaning up containers...${NC}"
docker-compose -f docker-compose-pipeline.yml down

echo ""
echo -e "${GREEN}Done!${NC}"
