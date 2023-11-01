import os

def main():
    run_number=3
    weights_path = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/case_studies/driverdrowsiness/pretrained/sub9/model.npy"
    approx_layers = 0
    eeg_test_data_path = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/delphi/rust/experiments/src/validation/compactCNN/Eeg_Samples_and_Validation"
    num_samples = 3
    accuracy_results_path = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/delphi/rust/experiments/src/validation/compactCNN/Eeg_Samples_and_Validation/Classification_Results{}.txt".format(run_number)
    output_file = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/delphi/rust/experiments/src/validation/compactCNN/validation_runs/validation_run{}.txt".format(run_number)
    os.chdir("/home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/delphi/rust/experiments/src/inference")
    os.system("cargo +nightly run --bin compact-cnn-sequential-inference -- --weights {} --layers {} --eeg_data {} --num_samples {} --results_file {} >> {}".format(weights_path, approx_layers, eeg_test_data_path, num_samples, accuracy_results_path, output_file))

if __name__ == "__main__":
    main()
