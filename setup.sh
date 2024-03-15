cp ./src/utils/format.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

cd ./models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd ..