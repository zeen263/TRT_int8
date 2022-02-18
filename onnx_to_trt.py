#!/usr/bin/env python3

# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import logging
import argparse

import tensorrt as trt

# local module
import processing
from calibrator import _Calibrator, get_calibration_files  


TRT_LOGGER = trt.Logger()
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Creates a TensorRT engine from the provided ONNX file.\n")
    parser.add_argument("--onnx", required=True,
                        help="The ONNX model file to convert to TensorRT")
    parser.add_argument("-o", "--output", type=str, default="model.engine",
                        help="The path at which to write the engine")
    parser.add_argument("-b", "--max_batch_size", type=int, default=32,
                        help="The max batch size for the TensorRT engine input")
    parser.add_argument("-v", "--verbosity", type=str,
                        help="Verbosity for logging. (None) for ERROR, (simple) for INFO/WARNING/ERROR, (verbose) for VERBOSE.")
    parser.add_argument("--explicit_batch", action='store_true',
                        help="Set trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH flag.")
    parser.add_argument("--fp16", action="store_true",
                        help="Attempt to use FP16 kernels when possible.")
    parser.add_argument("--int8", action="store_true",
                        help="Attempt to use INT8 kernels when possible. This should generally be used in addition to the \
                              --fp16 flag. ONLY SUPPORTS RESNET-LIKE MODELS SUCH AS RESNET50/VGG16/INCEPTION/etc.")
    parser.add_argument("--calibration_cache", default="calibration.cache",
                        help="(INT8 ONLY) The path to read/write from calibration cache.")
    parser.add_argument("--calibration_data", default=None,
                        help="(INT8 ONLY) The directory containing {*.jpg, *.jpeg, *.png} files to use for calibration. \
                              (ex: Imagenet Validation Set)")
    parser.add_argument("--calibration_batch_size", type=int, default=32,
                        help="(INT8 ONLY) The batch size to use during calibration.")
    parser.add_argument("--max_calibration_size", type=int, default=None,
                        help="(INT8 ONLY) The max number of data to calibrate on from --calibration_data.")
    parser.add_argument("-p", "--preprocess_func", type=str, default=None,
                        help="(INT8 ONLY) Function defined in 'processing.py' to use for pre-processing calibration data.")
    args, _ = parser.parse_known_args()

    # Adjust logging verbosity
    if args.verbosity is None:
        TRT_LOGGER.min_severity = trt.Logger.Severity.ERROR
    # -v
    elif args.verbosity == "simple":
        TRT_LOGGER.min_severity = trt.Logger.Severity.INFO
    # -vv
    elif args.verbosity == "verbose":
        TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE

    logger.info("TRT_LOGGER Verbosity: {:}".format(TRT_LOGGER.min_severity))

    network_flags = 0
    if args.explicit_batch:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(network_flags) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = args.max_batch_size
        builder.max_workspace_size = 2**30 * 5  # 5GiB

        if args.fp16:
            logger.info("Using FP16 build flag")
            builder.fp16_mode = True

        if args.int8:
            logger.info("Using INT8 build flag")
            builder.int8_mode = True

            # Use calibration cache if it exists
            if os.path.exists(args.calibration_cache):
                logger.info("Skipping calibration files, using calibration cache: {:}".format(args.calibration_cache))
                calibration_files = []
            # Use calibration files from validation dataset if no cache exists
            else:
                logger.info("Calib cache does not exist, doing calibration...")
                if not args.calibration_data:
                    raise ValueError("ERROR: Int8 mode requested, but no calibration data provided. "
                                     "Please provide --calibration_data /path/to/calibration/files")

                calibration_files = get_calibration_files(args.calibration_data, args.max_calibration_size)

            # Choose pre-processing function for INT8 calibration
            
            if args.preprocess_func is not None:
                preprocess_func = getattr(processing, args.preprocess_func)
            else:
                preprocess_func = processing.resize_ARGOS

            builder.int8_calibrator = _Calibrator(calibration_files=calibration_files,
                                                         batch_size=args.calibration_batch_size,
                                                         cache_file=args.calibration_cache,
                                                         preprocess_func=preprocess_func)

        # Fill network atrributes with information by parsing model
        with open(args.onnx, "rb") as f:
            parsed = parser.parse(f.read())

            # Exit if parsing fails
            if not parsed:
                logger.error("Failed to parse model.")
                sys.exit(1)

            # Exit if parsed network contains no layers
            if network.num_layers == 0:
                logger.error("Parsed network has 0 layers. Exiting...")
                sys.exit(1)

            # Mark output if not already marked
            if not network.get_output(0):
                logger.info("No output layer found, marking last layer as output. Correct this if wrong.")
                network.mark_output(network.get_layer(network.num_layers-1).get_output(0))


        logger.info("Building Engine...")
        with builder.build_cuda_engine(network) as engine, open(args.output, "wb") as f:
            logger.info("Writing engine to {:}".format(args.output))
            f.write(engine.serialize())
        
        logger.info(args.calibration_cache)


if __name__ == "__main__":
    main()
