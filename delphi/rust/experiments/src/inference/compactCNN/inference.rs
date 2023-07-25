// clap is a rust framework which allows you to easily
// implement argument parsing for your program

use clap::{App, Arg, ArgMatches};
use experiments::compact_cnn::construct_compact_cnn;
use neural_network::{ndarray::Array4, npy::NpyData};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::{io::Read, path::Path};

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn get_args() -> ArgMatches<'static> {
    App::new("compact-cnn-inference")
        .arg(
            Arg::with_name("weights")
                .short("w")
                .long("weights")
                .takes_value(true)
                .help("Path to weights")
                .required(true),
        )
        .arg(
            Arg::with_name("layers")
                .short("l")
                .long("layers")
                .takes_value(true)
                .help("Number of polynomial layers (0-7)")
                .required(true),
        )
        .get_matches()
}

fn main() {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let args = get_args();
    let weights = args.value_of("weights").unwrap();
    let layers = clap::value_t!(args.value_of("layers"), usize).unwrap();

    let mut network = construct_compact_cnn(None, 1, layers, &mut rng);
    let architecture = (&network).into();

    // load network weights
    network.from_numpy(&weights).unwrap();

    // Open EEG data and class
    let mut buf = vec![];
    std::fs::File::open(Path::new(""))


}