
datasets=(Gun_Point FaceFour Lighting2 Car Beef Coffee Plane BeetleFly BirdChicken Arrowhead Herring Lighting7)

for dataset in "${datasets[@]}"
do
    python triplet_siamese_mlp.py $dataset >> output.txt
done

