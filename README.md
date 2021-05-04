# EE 5450 Module 3 Homework 0

In this homework, you will practice using the OpenCV image processing
tools to find a Pokemon (either a Psyduck or a Pikachu).  You'll notice two 
Python files: `segmenter.py` and `test_segmenter.py`.  The former contains
the class definitions and the latter contains sample tests.  There are four
supplied test images in the `samples/` directory, of which are all used for
testing.  Note that you would normally have a training and testing data set,
but we are just practicing the image processing tools here and will explore
machine learning over the next lessons and assignment.

To get started:

1. I have used an interface-based design for this particular
   assignment.  `SegmenterInterface` is a class that requires its derived
   classes (meaning class that implement the interface) to implement its 
   methods.  The gist of each method is written as docstrings.  Because the 
   base interface has docstrings, you can just copy/paste the docstrings 
   over to your implementations of those functions.
   
2. Implement the `PikachuSegmenter` class first.  This is simpler because
   the test images contains only one Pikachu.  I recommend 
   to tmplement the functions in order (feel free to implement any helper 
   functions needed):
   
    a. `enhance_image()` to perform CLAHE on the each channel.
   
    b. `threshold_enhanced_image()` to get a noisy thresholded image of the
       object(s) you are looking for.
   
    c. `clean_thresholded_image()` to clean up your noisy thresholded image

    d. `get_combined_thresholded_image()` to reduce the four-channel binary 
       image to a single binary image
   
    e. `get_bounding_boxes()` to call all of these functions and spit out 
       the bounding boxes (list of tuple of: x_left, y_top, width, height) 
       that contain the object desired.
   
3. Once you have the `PikachuSegmenter` working, try your hand at the `PsyduckSegmenter`.

## Fair Use Notice

Please don't sue me, Pokemon Company International!  I think this is considered
fair use since it's for educational teaching purposes.
