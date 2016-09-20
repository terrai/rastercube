import numpy as np
from sklearn.metrics import classification_report


def hansen_index_to_year(index):
    return 2000 + (index)


def year_to_hansen_index(year):
    return (year - 2000)


def statistic_to_percent(value, total_number):
    try:
        return value * 100.0 / total_number
    except ZeroDivisionError:
        return np.nan


def compute_statistics(validation, detection, mask, stupid=False):
    """
    Takes two boolean 2d-arrays and computes sensitivity (TPR),
    specificity (TNR), precision (PPV), accuracy (ACC) and F1-score.
    """
    assert validation.shape == detection.shape, "Shapes do not match!"

    # Only considers the first detection as a valid one
    # Dismiss the following ones on the same pixel
    if not stupid and (len(detection.shape) > 2):
        reduce_prev = np.logical_or.reduce
        for date in range(1, detection.shape[2]):
            detections_before = reduce_prev(detection[:, :, 0:date],
                                            axis=2)
            common = np.logical_and(detections_before, detection[:, :, date])
            assert not np.any(common), "You have events detected multiple" +\
                                       "times in the detection array!"
            validations_before = reduce_prev(validation[:, :, 0:date],
                                             axis=2)
            common = np.logical_and(validations_before, validation[:, :, date])
            assert not np.any(common), "You have events detected multiple" +\
                                       "times in the validation array!"

    validation[~mask] = False
    validation_vector = validation.flatten()
    detection[~mask] = False
    detection_vector = detection.flatten()

    labels = [False, True]
    target_names = ['Untouched', 'Deforested']

    result1 = classification_report(validation_vector,
                                    detection_vector,
                                    digits=5,
                                    labels=labels,
                                    target_names=target_names)

    if len(validation.shape) > 2:
        result = "Per year comparison:\n" + result1
        validation_vector = (np.logical_or.reduce(validation,
                                                  axis=2)).flatten()
        detection_vector = (np.logical_or.reduce(detection, axis=2)).flatten()

        result2 = classification_report(validation_vector,
                                        detection_vector,
                                        digits=5,
                                        labels=labels,
                                        target_names=target_names)

        result += "\nFlattened comparison:\n" + result2
    else:
        result = result1

    return result
