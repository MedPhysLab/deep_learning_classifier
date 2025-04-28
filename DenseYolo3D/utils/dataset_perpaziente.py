import os
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.utils import resample
import numpy as np
import pydicom as dicom
import imageio
from PIL import Image
import numpy as np
import io
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from typing import AnyStr, BinaryIO, Dict, List, NamedTuple, Optional, Union
from joblib import Parallel,delayed
class DataProcessor:
    """
    A class to handle data loading, preprocessing, and splitting for the dataset.
    """

    def __init__(self, root_path,aux_path,df= None):
        """
        Initialize the DataProcessor.

        Args:
            root_path (str): root directory.
        """
        self.root_path = root_path
        self.aux_path= aux_path
        self.name_files = []
        self.df = df
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.ribilanciato = 0
        self.filtered = None
        self.pr_filter_index =  []
        self.file_paths=[]
        self.unique_fl= None
        self.dont_print = ["Pixel Data", "File Meta Information Version"]




    def preprocess_aux_files(self, auxiliary_data_file_names=["dataset DUKE training.ods", "BCS-DBT-boxes-train-v2.csv","SELECTED PATIENTS.ods"]):

        # Define the paths to the files
        self.file_paths = [self.aux_path + file_name for file_name in auxiliary_data_file_names]
        #print(self.file_paths)
        print(self.file_paths) 

        # Read the files into dataframes
        df_ods = pd.read_excel(self.file_paths[0], engine='odf')
        df_boxes = pd.read_csv(self.file_paths[1])
        dfs_selected_patients = pd.read_excel(self.file_paths[2],sheet_name=None, engine='odf')
        #print(dfs_selected_patients.keys())
        df_selected_patients_cancer = dfs_selected_patients["CANCER"]
        df_selected_patients_benign = dfs_selected_patients["BENIGN"]
        df_selected_patients_sane = dfs_selected_patients["NORMAL"]

        df_selected_patients = pd.concat([df_selected_patients_benign,df_selected_patients_cancer,df_selected_patients_sane]);

        df_selected_patients["name_file"]= df_selected_patients["PatientID"]+"-"+df_selected_patients["View"].str.upper()+".dcm";

        df_selected_patients.drop_duplicates(["name_file"],inplace=True)
        df_ods["name_file"]= df_ods["PatientID"]+ "-"+df_ods["View"].str.upper()+".dcm"
        df_boxes["name_file"]= df_boxes["PatientID"]+ "-"+df_boxes["View"].str.upper()+ ".dcm"
        merged_df= df_ods.merge(df_boxes , on='name_file',how="left").merge(df_selected_patients,on="name_file",how="inner")
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_y')]
        merged_df.columns = merged_df.columns.str.replace('_x$', '', regex=True)

        self.df = merged_df
        self.df = self.df.loc[:, ~self.df.columns.duplicated(keep="first")]
        return self.df

    def load_data_paths(self):
        # store all the paths of the files in each subfolder in the folders in the list_of_folders 
        paths = []
        for root, _, files in os.walk(self.root_path):
            for file in files:
                if file.endswith(".dcm"):
                    paths.append(os.path.join(root, file))




        #extract the name of the files from the path
        self.name_files = [os.path.basename(path) for path in paths]
        #print(self.name_files)

        # create a DataFrame from the paths
        self.df_paths = pd.DataFrame(self.name_files, columns=[ 'name_file'])
        self.df_paths['path'] = paths



        return self.df,self.df_paths



    def filter_data(self,filter_columns):
        """
        Preprocess the DataFrame by filtering rows. 
        """
        # keep only the rows with empty cells in the columns listed in filter_columns
        for el in filter_columns:
            self.pr_filter_index.extend(self.df[pd.isna(self.df[el])].index.tolist())



        if self.pr_filter_index is not None:
            self.df = self.df_paths.merge(self.df.loc[self.pr_filter_index],on = "name_file",how = "inner")
            #self.df_paths=self.df_paths[~self.df_paths["name_file"].isin(self.pr_filter)]

        self.df = self.df.loc[:, ~self.df.columns.duplicated(keep="first")]

        self.unique_fl = self.df.drop_duplicates(["name_file"])

        print("Beware, it still contains duplicates due to the info on the slices. For a dataframe with unique elements access unique_fl through the class")

        return self.df




    def split_data_train_val(self,column_name = "PatientID", test_size=0.2, random_state= 14):
        """
        Split the data into training and validation sets using GroupShuffleSplit so that patients are not mixed.

        Args:
            test_size (float): Proportion of the dataset to include in the validation split.
            random_state (int): Random seed for reproducibility.
            nome_colonna(str): nome della colonna da usare per lo splitting
        """
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
        split = splitter.split(self.unique_fl, groups=self.unique_fl[column_name])
        train_inds, val_inds = next(split)

        #return train_inds, val_inds

        self.train_df = self.unique_fl.iloc[train_inds]
        self.val_df = self.unique_fl.iloc[val_inds]
        print(f" after split train shape {self.train_df.shape}")
        print(f" after split val shape {self.val_df.shape}")
        print("Attenzione che i pazienti non sono ancora bilanciati tra sani e malati.")
        return self.train_df, self.val_df

    def balance_data(self, ribilanciare=1):
        """
        Balance the training and validation datasets by downsampling the majority class.

        Args:
            ribilanciare (int): Flag to indicate whether to balance the data (1 for yes, 0 for no).
        """
        if ribilanciare == 1:
            # Balance training data
            tumor_class_count_train = self.train_df['Malata'].value_counts()
            majority_class_train = 1 if tumor_class_count_train[1] > tumor_class_count_train[0] else 0
            minority_class_train = 1 - majority_class_train

            majority_df_train = self.train_df[self.train_df['Malata'] == majority_class_train]
            minority_df_train = self.train_df[self.train_df['Malata'] == minority_class_train]

            downsampled_majority_df_train = resample(
                majority_df_train, replace=False, n_samples=minority_df_train.shape[0], random_state=42
            )
            self.train_df = pd.concat([downsampled_majority_df_train, minority_df_train]).sort_values(by=['PatientID', 'Slice'])

            # Balance validation data
            tumor_class_count_val = self.val_df['Malata'].value_counts()
            majority_class_val = 1 if tumor_class_count_val[1] > tumor_class_count_val[0] else 0
            minority_class_val = 1 - majority_class_val

            majority_df_val = self.val_df[self.val_df['Malata'] == majority_class_val]
            minority_df_val = self.val_df[self.val_df['Malata'] == minority_class_val]

            downsampled_majority_df_val = resample(
                majority_df_val, replace=False, n_samples=minority_df_val.shape[0], random_state=42
            )
            self.val_df = pd.concat([downsampled_majority_df_val, minority_df_val]).sort_values(by=['PatientID', 'Slice'])

            self.ribilanciato = 1
            print(f" after bilanciamento train shape {self.train_df.shape}")
            print(f" after bilanciamento val shape {self.val_df.shape}")
            

            return self.train_df, self.val_df

    def split_validation_test(self, test_size=0.5, random_state=42):
        """
        Split the validation set into validation and test sets.

        Args:
            test_size (float): Proportion of the validation set to include in the test split.
            random_state (int): Random seed for reproducibility.
        """
        self.val_df, self.test_df = train_test_split(
            self.val_df, test_size=test_size, stratify=self.val_df["Malata"], random_state=random_state
        )

        return self.test_df, self.val_df


    def get_meta_data(self, row):
        """
        Get the meta data contained in the dcm file

        Input:
            row: the row of the dataframe containing the path to the file in the column "path".

        Returns:
            printed text of the metadata.
        """
        if row is None:
            raise ValueError("The provided row is None.")
        
        dicom_path = row["path"] 
        ds = dicom.dcmread(dicom_path)  # Read the .dcm file
        self._print_metadata(ds)


    #private method should not be called by the users
    def _print_metadata(self, ds, indent=0):
        """Recursively print metadata with indentation."""
        indent_string = "   " * indent
        next_indent_string = "   " * (indent + 1)



        for elem in ds:
            if elem.VR == "SQ":  # A sequence
                print(indent_string, elem.name)
                for sequence_item in elem.value:
                    self._print_metadata(sequence_item, indent + 1)
                    print(next_indent_string + "---------")
            else:
                if elem.name in self.dont_print:
                    print("""<item not printed -- in the "don't print" list>""")
                else:

                    repr_value = repr(elem.value)
                    if len(repr_value) > 50:
                        repr_value = repr_value[:50] + "..."
                    print(f"{indent_string} {elem.name} = {repr_value}")   

        return 0
    
    def _get_dicom_laterality(ds: dicom.dataset.FileDataset) -> str:
        """Unreliable - DICOM laterality is incorrect for some cases"""
        return ds[0x5200, 0x9229][0][0x0020, 0x9071][0][0x0020, 0x9072].value

    
    def _get_image_laterality(self,pixel_array: np.ndarray) -> str:
        """
        Determine the laterality of an image based on the sum of pixel values 
        at the left and right edges of the image.
        Args:
            pixel_array (np.ndarray): A 2D numpy array representing the image.
        Returns:
            str: 'R' if the sum of pixel values on the left edge is less than 
                 the sum on the right edge, otherwise 'L'.
        """

        left_edge = np.sum(pixel_array[:, 0])  # sum of left edge pixels
        right_edge = np.sum(pixel_array[:, -1])  # sum of right edge pixels
        return "R" if left_edge < right_edge else "L"
    


    def animated_gif(self, row, step=4,target_height=640,duration= 140,save = False):
        """
        Generates a list of images from a DICOM file to create an animated GIF coded in int8 for a quick look.
        Args:
            row (dict): A dictionary containing metadata about the DICOM file, including the file path.
            step (int, optional): The step size for selecting frames from the DICOM file. Defaults to 2.
            target_height (int, optional): The target height of the resized frames. Defaults to 640.
            duration (int, optional): The duration for each frame in the GIF. Defaults to 140 milliseconds.
            save (Bol,optional): save the gif. default false.
        Returns:
            list: A list of PIL Image objects representing the frames of the animated GIF.


        Example:
        from IPython.display import Image as ipyimage
        gif_io = classe_dati.animated_gif(train_df.iloc[22])
        ipyimage(data=gif_io.read(), format='gif')
        """
        # Create a BytesIO object to store the GIF in memory
        gif_io = io.BytesIO()
        images= []
        dicom_path = row["path"] 
        ds = dicom.dcmread(dicom_path)  # Read the .dcm file
        max_frames = ds.NumberOfFrames
        global_min = ds.pixel_array.min()
        global_max = ds.pixel_array.max()
        # Normalize the pixel values to the range 0-255 and convert to uint8
        arrays = [((ds.pixel_array[el, :, :] - global_min) / (global_max - global_min) * 255).astype(np.uint8)
          for el in range(0, max_frames, 2)]
        #arrays =  [ds.pixel_array[el,:,:].astype(np.uint8)   for el in range(0,max_frames,2)  ]
        for array in arrays:
            original_height,original_width = array.shape
            target_width= int(target_height * original_width/original_height)
            image= Image.fromarray(array)
            image = image.resize((target_height,target_width))

            images.append(image  )

    
        imageio.mimsave(gif_io,images,duration=duration,format="GIF",loop=0)
        gif_io.seek(0)

        if save:
            imageio.mimsave(f"{row['name_file']}.gif", images)


        return gif_io
    



    def _get_window_center(self, ds: dicom.dataset.FileDataset) -> np.float32:
        return np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1050].value)


    def _get_window_width(self, ds: dicom.dataset.FileDataset) -> np.float32:
        return np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1051].value)




    def decompress_and_flip_image(self,row,index =0):
        """
        Decompresses a DICOM image, checks its laterality, and flips it if necessary.
        Args:
            row (pd.Series): A pandas Series containing the DICOM file path and view laterality.
            index (int, optional): Index of the image slice to process. Defaults to 0.
        Returns:
            np.ndarray: The possibly flipped pixel array of the DICOM image.
        """

        dcm_path,view_laterality= row['path'],row['View'].upper()[0]
        ds = dicom.dcmread(dcm_path)
        if 'PixelData' in ds:
            pixel_array = ds.pixel_array
        image_laterality = self._get_image_laterality(pixel_array[index])
        print(f" image laterality:... {image_laterality}")
        print(f"view laterality:...  {view_laterality}" )
        if not image_laterality == view_laterality:
            pixel_array = np.flip(pixel_array, axis=(-1, -2))


        window_center = self._get_window_center(ds)
        window_width = self._get_window_width(ds)
        low = (2 * window_center - window_width) / 2
        high = (2 * window_center + window_width) / 2
        pixel_array = rescale_intensity(
            pixel_array, in_range=(low, high), out_range="dtype"
        )


        return pixel_array
    


    def decompress_and_save_dataset(self,df,output_root_path=""):
        """
        Decompresses and saves a dataset of images.
        This function iterates over each row in the provided DataFrame, decompresses and flips the image,
        and then saves the processed image to the specified output path. If no output path is provided,
        the image is saved to a default directory within the root path.
        Args:
            df (pandas.DataFrame): The DataFrame containing image data. Each row should have 
                                            a 'name_file' column with the image filename.
            output_path (str, optional): The directory where the processed images will be saved. 
                                         Defaults to an empty string, which uses the default directory.

        Returns:
            int: Returns 1 if the DataFrame is None, indicating that a DataFrame is necessary.
        Raises:
            OSError: If there is an error creating the output directory or saving the image.
        Notes:
            - The function assumes that the 'decompress_and_flip_image' method is defined elsewhere in the class.
            - The images are saved in TIFF format with deflate compression.
        """



        if df is None:
                print("df necessario")

                return 1
        for index, row in df.iterrows():
            pixel_array = self.decompress_and_flip_image(row)
            base_name = os.path.splitext(row["name_file"])[0]
            print(base_name)
            # Save the decompressed and possibly flipped image
            if output_root_path == "":
                output_path = os.path.join(self.root_path, "processed_images/train/", base_name +".tiff")
            else:
                output_path = os.path.join(output_root_path, base_name +".tiff")


            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            image = Image.fromarray(pixel_array[0])
            images = [Image.fromarray(slice_) for slice_ in pixel_array[1:]]
            image.save(output_path,save_all= True, append_images=images, compression="tiff_deflate")




            #imageio.imwrite(save_path.replace(".dcm", ".tiff"), pixel_array)
    
            # Save the flipped image back to the DICOM file
            #ds.PixelData = flipped_image.tobytes()
            #ds.save_as(dcm_path)



    def get_data(self, df):  
        """
        Processes the given DataFrame and yields the pixel array of each row.
        Args:
            df (pandas.DataFrame): The DataFrame containing the data to be processed.
        Yields:
            
        Raises:
            ValueError: If the DataFrame is None.
        """
        #if df is None:
            #raise ValueError("df necessario")
        #for index, row in df.iterrows():
            #print(f"Iteration number: {index}")
            #print(f"filename: {row['name_file']}")
            #pixel_array = self.decompress_and_flip_image(row)
            #yield pixel_array, row
        if df is None:
            raise ValueError("df necessario")
        for index, row in df.iterrows():
            print(f"Iteration number: {index}")
            print(f"filename: {row['name_file']}")
    



            yield index, row




