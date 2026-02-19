# Enabling Google Cloud Text-to-Speech API & Service Account Permissions

## 1. Verify Project

Ensure you are in the correct project: **`ondertitels-486017`**
(Project Number: `667654992936`)

## 2. Enable the API

1. Visit the API Library page directly for this project:  
   [https://console.cloud.google.com/apis/library/texttospeech.googleapis.com?project=ondertitels-486017](https://console.cloud.google.com/apis/library/texttospeech.googleapis.com?project=ondertitels-486017)
2. Click **ENABLE**.
3. If it says "Manage", it is already enabled. Try disabling and re-enabling it if it's stuck.

## 3. Verify Service Account Permissions

Sometimes the Service Account needs specific roles.

1. Go to **IAM & Admin** > **IAM**.
2. Find the service account email (it's inside your JSON key file, usually looking like `...@ondertitels-486017.iam.gserviceaccount.com`).
3. Click the **Pencil icon** (Edit) for that user.
4. Ensure it has one of the following roles:
   - **Owner** (easiest for dev)
   - **Editor**
   - **Cloud Text-to-Speech API User**
5. Save.

## 4. Forced Propagation

Sometimes forcing a refresh helps:

- Use the `gcloud` CLI if you have it installed:

  ```bash
  gcloud services enable texttospeech.googleapis.com --project ondertitels-486017
  ```
