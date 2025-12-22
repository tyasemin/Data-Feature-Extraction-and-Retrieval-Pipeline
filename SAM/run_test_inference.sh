#!/bin/bash
# Run SAM inference testing on 20 random images

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   SAM Inference Testing Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in the SAM directory
if [ ! -f "docker-compose-test.yml" ]; then
    echo -e "${YELLOW}Error: docker-compose-test.yml not found!${NC}"
    echo "Please run this script from the SAM directory"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p sam_inference
echo -e "${GREEN}✓${NC} Created output directory: ./sam_inference"

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

# Run the test
echo ""
echo -e "${BLUE}Running SAM inference test...${NC}"
echo -e "${YELLOW}This will process 20 randomly selected images${NC}"
echo ""

docker-compose -f docker-compose-test.yml up --abort-on-container-exit

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   Test completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Results saved to: ${BLUE}./sam_inference/${NC}"
    echo ""
    
    # Count output files
    num_files=$(ls -1 sam_inference/*.png 2>/dev/null | wc -l)
    echo -e "Generated ${GREEN}${num_files}${NC} output images"
    echo ""
    echo "To view results:"
    echo "  ls -lh sam_inference/"
else
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}   Test encountered errors${NC}"
    echo -e "${YELLOW}========================================${NC}"
fi

# Cleanup
echo ""
echo -e "${BLUE}Cleaning up containers...${NC}"
docker-compose -f docker-compose-test.yml down

echo ""
echo -e "${GREEN}Done!${NC}"
