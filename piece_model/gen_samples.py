import os
import random
from pathlib import Path
import sys
import subprocess

NUM_ITER=4
OUTPUT_DIR="../piece_samples"
BLACK="black"
WHITE="white"
DIR_OPTIONS=["NorthWest","North","NorthEast","West","East","SouthWest","South","SouthEast"]
TEXT_OPTIONS=["a","b","c","d","e","f","g","h","A","B","C","D","E","F","G","H","1","2","3","4","5","6","7","8","1","2","3","4","5","6","7","8"]
FONTS=["AvantGarde","Bookman","Courier","Helvetica","NewCenturySchlbk","Palatino","Symbol","Times"]
BRIGHTNESS_DELTA=25

ONLY_PIECE=""

for root, dirs, files in os.walk('../piece_files/'):
  for file in files:
    if file.endswith(".png"):
      path = os.path.join(root, file)
      collection = os.path.split(os.path.dirname(path))[1]
      basename = os.path.basename(path)
      basename_no_extn = basename.removesuffix(".png")
      output_dir = os.path.join(OUTPUT_DIR, basename_no_extn)
      if ONLY_PIECE and basename_no_extn != ONLY_PIECE:
        continue
      print(path)
      for i in range(NUM_ITER if basename_no_extn != 'nothing' else 20):
        offset_x = random.randint(-10, 10)
        offset_y = random.randint(-10, 10)
        scale = random.uniform(.8,1.2)
        shear = random.randint(-10, 10)
        rotation = random.randint(-10, 10)
        r=random.randint(0, 255)
        g=random.randint(0, 255)
        b=random.randint(0, 255)
        color=f"'rgb({r},{g},{b})'"
        text_dir=DIR_OPTIONS[random.randrange(len(DIR_OPTIONS))]
        text_content=TEXT_OPTIONS[random.randrange(len(TEXT_OPTIONS))]
        font=FONTS[random.randrange(len(FONTS))]
        text_rot=random.randint(-10,10)
        text_scale=random.uniform(.8,1.5)
        draw_x1 = random.randrange(11)
        draw_y1 = random.randrange(11)
        draw_x2 = random.randrange(11)
        draw_y2 = random.randrange(11)
        line_x = random.randrange(97)
        line_y = random.randrange(97)
        draw_color=f"'rgb({b},{r},{r})'"
        brightness_func=lambda:random.randint(100-BRIGHTNESS_DELTA,100+BRIGHTNESS_DELTA)
        darker=random.randint(100-BRIGHTNESS_DELTA,100)
        lighter=random.randint(100,100+BRIGHTNESS_DELTA)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_dir, f"{collection}_{i}_vanilla_{file}")
        subprocess.run(f"gm convert {path} -fill {color} -background {color} -crop 96x96 -resize '96x96!' -extent 0x0 {output_file}", capture_output=True, check=True, shell=True)
        output_file = os.path.join(output_dir, f"{collection}_{i}_light_{file}")
        subprocess.run(f"gm convert {path} -fill {color} -background {color} -crop 96x96 -resize '96x96!' -extent 0x0 -modulate {lighter} {output_file}", capture_output=True, check=True, shell=True)
        output_file = os.path.join(output_dir, f"{collection}_{i}_dark_{file}")
        subprocess.run(f"gm convert {path} -fill {color} -background {color} -crop 96x96 -resize '96x96!' -extent 0x0 -modulate {darker} {output_file}", capture_output=True, check=True, shell=True)
        output_file = os.path.join(output_dir, f"{collection}_{i}_straight_{file}")
        subprocess.run(f"gm convert {path} -fill {color} -background {color} -affine {scale},0,0,{scale},{offset_x},{offset_y} -transform -crop 96x96 -resize '96x96!' -extent 0x0 -modulate {brightness_func()} {output_file}", capture_output=True, check=True, shell=True)
        if basename_no_extn != 'nothing':
          output_file = os.path.join(output_dir, f"{collection}_{i}_rot_{file}")
          subprocess.run(f"gm convert {path} -fill {color} -background {color} -rotate {rotation} -affine {scale},0,0,{scale},{offset_x},{offset_y} -transform -crop 96x96 -resize '96x96!' -extent 0x0 -modulate {brightness_func()} {output_file}", capture_output=True, check=True, shell=True)
          output_file = os.path.join(output_dir, f"{collection}_{i}_shear_{file}")
          subprocess.run(f"gm convert {path} -fill {color} -background {color} -shear {shear}x{shear} -crop 96x96 -resize '96x96!' -extent 0x0 -modulate {brightness_func()} {output_file}", capture_output=True, check=True, shell=True)
        output_file = os.path.join(output_dir, f"{collection}_{i}_text_{file}")
        subprocess.run(f"gm convert {path} -fill {color} -background {color} -crop 96x96 -resize '96x96!' -extent 0x0 -fill {draw_color} -font {font} -draw \"gravity {text_dir} Scale {text_scale},{text_scale} text {draw_x1},{draw_y1} '{text_content}'\" -modulate {brightness_func()} {output_file}", capture_output=True, check=True, shell=True)
        output_file = os.path.join(output_dir, f"{collection}_{i}_textwarped_{file}")
        subprocess.run(f"gm convert {path} -fill {color} -background {color} -crop 96x96 -resize '96x96!' -extent 0x0 -fill {draw_color} -font {font} -draw \"gravity {text_dir} Rotate {text_rot} Scale {text_scale},{text_scale} text {draw_x2},{draw_y2} '{text_content}'\" -modulate {brightness_func()} {output_file}", capture_output=True, check=True, shell=True)
        output_file = os.path.join(output_dir, f"{collection}_{i}_line_vert_{file}")
        subprocess.run(f"gm convert {path} -fill {color} -background {color} -crop 96x96 -resize '96x96!' -extent 0x0 -fill {draw_color} -draw 'line {line_x},0,{line_x},96' -modulate {brightness_func()} {output_file}", capture_output=True, check=True, shell=True)
        output_file = os.path.join(output_dir, f"{collection}_{i}_line_horiz_{file}")
        subprocess.run(f"gm convert {path} -fill {color} -background {color} -crop 96x96 -resize '96x96!' -extent 0x0 -fill {draw_color} -draw 'line 0,{line_y},96,{line_y}' -modulate {brightness_func()} {output_file}", capture_output=True, check=True, shell=True)
        output_file = os.path.join(output_dir, f"{collection}_{i}_noisy_{file}")
        subprocess.run(f"gm convert {path} -fill {color} -background {color} -crop 96x96 -resize '96x96!' -extent 0x0 +noise Laplacian -modulate {brightness_func()} {output_file}", capture_output=True, check=True, shell=True)
        color=WHITE
        output_file = os.path.join(output_dir, f"{collection}_{i}_white_background_{file}")
        subprocess.run(f"gm convert {path} -fill {color} -background {color} -crop 96x96 -resize '96x96!' -extent 0x0 -modulate {brightness_func()} {output_file}", capture_output=True, check=True, shell=True)
        color=BLACK
        output_file = os.path.join(output_dir, f"{collection}_{i}_black_background_{file}")
        subprocess.run(f"gm convert {path} -fill {color} -background {color} -crop 96x96 -resize '96x96!' -extent 0x0 -modulate {brightness_func()} {output_file}", capture_output=True, check=True, shell=True)

"""
NUM_ITER=3
OUTPUT_DIR="../piece_samples"
#arr=(0 0 0 0 0 0 0 0 0 0 0)
#for i in $(seq 1000); do val=$(randbool); arr["$val"]=$((arr[val] + 1)) ; done
#echo "${arr[@]}"
for file in os.w:
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
    gm convert "$file" -fill "$color" -background "$color" -crop 96x96 -resize 96x96! -extent 0x0 +noise Laplacian "$OUTPUT_DIR/$basename_no_extn/${set}_${iter}_vanilla_$basename"
    gm convert "$file" -fill "$color" -background "$color" -affine "$scale,0,0,$scale,$offset_x,$offset_y" -transform -crop 96x96 -resize 96x96! -extent 0x0 +noise Laplacian "$OUTPUT_DIR/$basename_no_extn/${set}_${iter}_straight_$basename"
    gm convert "$file" -fill "$color" -background "$color" -rotate "$rotation" -affine "$scale,0,0,$scale,$offset_x,$offset_y" -transform -crop 96x96 -resize 96x96! -extent 0x0 +noise Laplacian "$OUTPUT_DIR/$basename_no_extn/${set}_${iter}_rot_$basename"
    gm convert "$file" -fill "$color" -background "$color" -shear "${shear}x${shear}" -crop 96x96 -resize 96x96! -extent 0x0 +noise Laplacian "$OUTPUT_DIR/$basename_no_extn/${set}_${iter}_shear_$basename"
  done
  break
done
"""