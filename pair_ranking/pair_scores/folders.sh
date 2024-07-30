# Bash command to keep only processed genoms :

mkdir -p /path_to_final_folder/; for f in /home/thibaut/KEEP_80_v3/*.txt; do bn=$(basename "$f" .txt);
if [ -f "/path_to_former_folder/pairs/$bn.pickle" ]; then cp "$f" "/path_to_final_folder/$bn.txt";
cp "/path_to_former_folder/pairs/$bn.pickle" "/path_to_final_folder/$bn.pickle"; fi; done

# Move to appropriate folders

mkdir pairs
mv *.pkl pairs

mkdir annotations
mv *.txt annotations
