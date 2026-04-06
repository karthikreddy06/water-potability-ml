name: Bug Report
description: Report a bug or issue
title: "[BUG] "
labels: ["bug"]

body:
  - type: markdown
    attributes:
      value: |
        Thank you for reporting a bug! Please fill out the form below.

  - type: textarea
    id: description
    attributes:
      label: Description
      description: A clear description of what the bug is
      placeholder: What happened?
    validations:
      required: true

  - type: textarea
    id: reproduce
    attributes:
      label: Steps to Reproduce
      description: Steps to reproduce the behavior
      placeholder: |
        1. Run 'python src/train_v2.py'
        2. Click on 'Batch Prediction'
        3. Upload CSV file
      value: |
        1.
        2.
        3.

  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: What should happen?
      placeholder: The model should train successfully

  - type: textarea
    id: actual
    attributes:
      label: Actual Behavior
      description: What actually happened?
      placeholder: Error message or unexpected result

  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: Your system information
      placeholder: |
        OS: Windows 10
        Python: 3.8.0
        Streamlit: 1.0.0

  - type: textarea
    id: logs
    attributes:
      label: Relevant Logs
      description: Any error logs or stack traces
      render: python

  - type: textarea
    id: additional
    attributes:
      label: Additional Context
      description: Any other context?
