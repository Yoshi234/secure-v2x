# V2V-Delphi-Applications
---
## Repository contains code for secure 2-party inference schemes of the following applications (using Delphi)
---
1. (Private) Driver-Drowsiness Detection
2. (Private) Red-Light Traffic Rule Violation
3. (Private) Traffic Flow Prediction and Optimization

## TODO: 8/5/24

1. Add the modified version of CrypTen to the repository with reproducible instructions
   to install the updated version of the code from source.
2. Clean up the crypten compactcnn implementation and convert to a script which can be run 
   more conveniently
3. Include instructions for accessing the data utilized for this work (make sure this is robust)
4. Update fully-automated RLR detection script, and move into the main v2x-delphi-2pc repo
   (the repo needs to be renamed as well to "v2x-2pc" or something)

For this code to work, we need to run all scripts from the case_studies package as a 
relative call now. This is because I have restructured everything as a package format.