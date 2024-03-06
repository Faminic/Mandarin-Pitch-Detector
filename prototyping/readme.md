The files in here are to help with prototyping.


Please note that I already ran Microsoft Azure on as many files as I could (some files just kept returning error), then processed that data alongside previous human-tagging and saved it to all_metadata.json.



The file 'ensemble_method.ipynb' contains some sample code to help you test any methods you develop.


Make sure all of these files are in the same folder, and also download and unzip the dataset to the same folder. When you run ensemble_method.ipynb, it reads metadata from all_metadata.json, and also reads audio from ./Chinese_DPA/audio. Note that the folder should be renamed with the underscore (I uploaded it with a space instead).


ensemble_method.ipynb will import the methods from the other .py files. Most of the methods are in pitch_methods.py, but I separated peak_detection to a separate file because the code for that is much longer. 

If you have finished your own method, I suggest you add another .py file for your method, and edit ensemble_method to import your method, then add in the code for ensemble_method to calculate and graph your method. Since you can see it on the same graph as the rest of the existing methods, you can compare more easily. If you look at the code in ensemble_method.ipynb, you'll see that after using the imported methods to estimate pitch, the next cell sets up a dictionary of what are essentially [time, frequency, weight] tuplets, and then the visualise_results method I imported will iterate over the dictionary to plot over a spectrogram.

Please note that the name 'ensemble_method' is currently a misnomer, I have not actually developed the ensemble yet. For now it's just to let us visually compare methods. 
