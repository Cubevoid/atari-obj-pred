cd ./models

if [ ! -f sam_vit_b_01ec64.pth ]; then
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi

if [ ! -f FastSAM-s.pt ]; then
    wget 'https://drive.usercontent.google.com/download?id=10XmSj6mmpmRb8NhXbtiuO9cTTBwR_9SV&export=download&authuser=0' -O FastSAM-s.pt
fi

if [ ! -f FastSAM-x.pt ]; then
    wget 'https://drive.usercontent.google.com/download?id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv&export=download&authuser=0&confirm=t&uuid=c85cd8f1-cb2b-47b3-a975-9c7018883c25&at=APZUnTXCRQs0Ehl-0A0Mn3dDgHAY%3A1710194177485' -O FastSAM-x.pt
fi

cd ..