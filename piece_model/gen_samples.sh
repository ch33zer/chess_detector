rand() {
	# Inclusive
	local min=${1:-0}
	# Exclusive
	local max=${2:-1}
	local bias=${3:-0}
	local rand=$RANDOM
  #>&2 echo "rand: $rand"
	local max_rand=32767
	if (($bias == 1)); then bias=" - .5"; else bias=""; fi
	echo "$(bc -l <<< "$rand * (($max - $min) / $max_rand) + $min $bias")"
}

randint() {
	# inclusive
	local min=${1:-0}
	# exclusive
	local max=${2:-32767}
	local float="$(rand $min $max 1)"
	printf "%.0f\n" "$float"
  echo "$RANDOM" > /dev/null
}

randbool() {
	echo "$(randint 0 2)"
}

NUM_ITER=3
OUTPUT_DIR="../piece_samples"
#arr=(0 0 0 0 0 0 0 0 0 0 0)
#for i in $(seq 1000); do val=$(randbool); arr["$val"]=$((arr[val] + 1)) ; done
#echo "${arr[@]}"
for file in ../piece_files/**/*.png; do
	echo "$file"
	set="$(echo "$file" | cut -f3 -d/)"
	basename="$(basename "$file")"
	basename_no_extn="$(basename -s .png "$file")"
	for iter in $(seq $NUM_ITER); do
    #echo "iter $iter"
		#read -r w h <<< $(gm identify "$file" -format "%w %h")
		offset_x="$(randint -10 10)"
    #echo "offset_x $offset_x"
		offset_y="$(randint -10 10)"
    #echo "offset_y $offset_y"
		scale="$(rand .8 1.2)"
    #echo "scale $scale"
    shear="$(randint -10 10)"
    #echo "shear $shear"
		rotation="$(randint -10 10)"
    #echo "rotation $rotation"
    r="$(randint 0 255)"
    #echo "r $r"
    g="$(randint 0 255)"
    #echo "g $g"
    b="$(randint 0 255)"
    #echo "b $b"
    color="rgb($r,$g,$b)"
    #echo "color $color"
		mkdir "$OUTPUT_DIR/${basename_no_extn}" 2> /dev/null
		gm convert "$file" -fill "$color" -background "$color" -crop 64x64 -resize 64x64! -extent 0x0 +noise Laplacian "$OUTPUT_DIR/$basename_no_extn/${set}_${iter}_vanilla_$basename"
		gm convert "$file" -fill "$color" -background "$color" -affine "$scale,0,0,$scale,$offset_x,$offset_y" -transform -crop 64x64 -resize 64x64! -extent 0x0 +noise Laplacian "$OUTPUT_DIR/$basename_no_extn/${set}_${iter}_straight_$basename"
		gm convert "$file" -fill "$color" -background "$color" -rotate "$rotation" -affine "$scale,0,0,$scale,$offset_x,$offset_y" -transform -crop 64x64 -resize 64x64! -extent 0x0 +noise Laplacian "$OUTPUT_DIR/$basename_no_extn/${set}_${iter}_rot_$basename"
    gm convert "$file" -fill "$color" -background "$color" -shear "${shear}x${shear}" -crop 64x64 -resize 64x64! -extent 0x0 +noise Laplacian "$OUTPUT_DIR/$basename_no_extn/${set}_${iter}_shear_$basename"
	done
  break
done
