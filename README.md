-Modules: OpenCV3 with numpy and pickle.

-Instructions on installing opencv3.0.0 module (using anaconda python 3):

conda create -n opencv numpy scipy scikit-learn matplotlib python=3
source activate opencv
Might need this: (conda install -c trung tbb=4.3_20141023)
conda install -c jlaura opencv3=3.0.0

-Usage: Use left mouse click to drag a rectangle on your marker. Marker should be a unique color on the screen (i.e. only green object) and ideally will have a small surface area. Then use 'm' key to toggle drawing mode to start drawing on the screen with the marker. When you're done click 'r' to enter shell-command-typing-mode, type in your shell command and click enter to save the binding. Use 'd' after drawing a gesture to find the closest match of the saved gestures and execute the corresponding shell command.

-Code: OpenCV relies on constant loop so decided not to use object oriented design. Use global variables with methods that are called from the loop. Overloaded python print method to print to screen as well. Used pickle to save bindings to file as json dictionary so that bindings survive restarts. I save the canvas as an image (only the drawing) and then iterate through them in gesture detection to compare contours to pick the one closest to the current drawing. Use openCV matchShape function to compare the shapes in the gestures. Smaller return value means closer match. Iterate through all gesture image files, find the one with the smallest match value (and over the threshold I set), and use its filename as the dictionary key to execute the corresponding shell command previously coupled with it.

-Problems: opencv matchShapes method is sometimes less reliable when the shape is jittered, but works well when the drawing is smooth (smaller value means more similar).
