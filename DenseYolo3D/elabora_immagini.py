from utils import duke3d_dataprep as dataprep
from utils import dataset_perpaziente as dp
import datetime
import cProfile
import pstats

def elabora_immagini(root_path = "/srv/LongTerm01/pcaterino/Duke/Custom_Dataset", auxiliary_data_root_path = "/home/pcaterino/duke_aux/", auxiliary_data_file_names = ["dataset DUKE training.ods", "BCS-DBT-boxes-train-v2.csv", "SELECTED PATIENTS.ods"], random_state=34):


    # Prepare pandas dataframes
    classe_dati = dp.DataProcessor(root_path,auxiliary_data_root_path )
    classe_dati.preprocess_aux_files()
    df_tot, df_paths = classe_dati.load_data_paths()
    df = classe_dati.filter_data(["clips"])
    train_df, val_df = classe_dati.split_data_train_val(column_name="PatientID", random_state=random_state)
    train_df, val_df = classe_dati.balance_data()
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    train_data_generator = classe_dati.get_data(train_df)
    val_data_generator = classe_dati.get_data(val_df)

    # Save images
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    factor = 3
    root_output_folder = f"/home/pcaterino/3d_effort/resized/{now}/{factor}/"
    dataprep.save_images(root_output_folder, train_data_generator, split_type="train", factor=factor,classe=classe_dati)
    dataprep.save_images(root_output_folder, val_data_generator, split_type="val", factor=factor,classe= classe_dati)

if __name__ == "__main__":
    cProfile.run('elabora_immagini()', 'profile_output')
    p = pstats.Stats('profile_output')
    p.sort_stats('cumulative').print_stats(10)
    p.sort_stats('time').print_stats(10)
    p.sort_stats('calls').print_stats(10)
    p.sort_stats('tottime').print_stats(10)
    p.sort_stats('ncalls').print_stats(10)
    p.sort_stats('cumtime').print_stats(10)
