# Chess Detector

![santa_sample](https://github.com/ch33zer/chess_detector/blob/main/santa_sample.png)

OpenCV/Tensorflow project to detect chessboards in images, then identify the pieces present on the board. The idea was that you could use this to resume streamer/youtuber games on your own in Lichess.

This is entirely client side (ML model, OpenCV, etc), no server at all.

This is only a demo that sometimes works. If you want to do this for real use a commercial solution like [Chessvision.AI](https://chessvision.ai/)

## Use

https://blaise.gg/chess_detector/chess_detector.html


## Operation

You provide an image which gets drawn to a canvas. We use Open CV to convert it to black and white and run [Houghs edge detection](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html) on it through a surprisingly convulted process this eventually gives us a bounding box for each square on the board.

Once we have each square on the board we run a custom trained tensorflow model on each to give us the piece occupying that square.

From there we populate a [chessboard.js](https://chessboardjs.com/) instance to display the results.

While it sounds simple, theres a LOT going on here, mostly down to noisy images and things (cursors, overlays, etc) obscuring the image. The edge detection can also find the wrong area, which throws everything off. Accuracy definitely isn't perfect. 
