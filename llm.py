import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the YAML file & Extract the prompt text
with open("prompts.yaml", "r") as f:
    data = yaml.safe_load(f)
prompt_template = data["prompt_llm"]

# Load model
model_path = "/datas01/t0323469/checkpoints/Granite-4.0-micro"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


def evaluate_segmentation(fire_mask_data, smoke_mask_data, detected_objects_data, drone_metadata): 
    prompt = (prompt_template_targeted_atacks
        .replace("<<fire_mask_data>>", fire_mask_data)
        .replace("<<smoke_mask_data>>", smoke_mask_data)
        .replace("<<detected_objects_data>>", detected_objects_data)
        .replace("<<drone_metadata>>", drone_metadata)
    )

    messages = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=40)
    output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    output = output.replace("<|end_of_text|>", "")

    print(output)
    return output


fire_mask_data = """ [[1.32322207e-03 6.71209069e-04 4.72328044e-04 ... 4.02467384e-04
  5.52626967e-04 1.01990462e-03]
 [5.36409032e-04 1.37503244e-04 7.58531023e-05 ... 6.66956548e-05
  1.23400940e-04 4.13043541e-04]
 [2.92850833e-04 5.16843356e-05 3.03730594e-05 ... 2.36671094e-05
  5.06729775e-05 2.89957039e-04]
 ...
 [4.01892117e-04 1.01667640e-04 5.47963355e-05 ... 1.94544744e-04
  3.04792542e-04 7.22075230e-04]
 [5.54066035e-04 1.72799279e-04 1.14654795e-04 ... 3.60259321e-04
  5.14989079e-04 1.05691957e-03]
 [9.10344941e-04 4.00419260e-04 3.27457528e-04 ... 9.67846368e-04
  1.16272934e-03 1.75606750e-03]] """
smoke_mask_data = """ [[0.08184605 0.09032622 0.10243438 ... 0.04341868 0.04767516 0.06089932]
 [0.07976367 0.08616382 0.10049693 ... 0.02653459 0.03125579 0.04725264]
 [0.08144865 0.08832669 0.1030283  ... 0.02121554 0.02523499 0.04158044]
 ...
 [0.03778588 0.02517505 0.02076937 ... 0.0910183  0.08523472 0.08442432]
 [0.04452808 0.0330639  0.02938283 ... 0.08405705 0.07951685 0.07935051]
 [0.05427864 0.04320882 0.04017783 ... 0.06983933 0.06905077 0.07305752]] """
detected_objects_data = """ [{'type': 'car', 'bbox': (2451.3099365234375, 0.0, 2474.7364501953125, 18.46479034423828), 'score': 0.604515016078949}, {'type': 'car', 'bbox': (2226.9666748046875, 173.0142364501953, 2238.6124877929688, 188.4536895751953), 'score': 0.33422115445137024}, {'type': 'car', 'bbox': (349.1172790527344, 1163.2674560546875, 355.5012512207031, 1173.30615234375), 'score': 0.3412432372570038}, {'type': 'car', 'bbox': (850.6202850341797, 1223.4472351074219, 861.6568603515625, 1239.3555603027344), 'score': 0.6429840922355652}, {'type': 'car', 'bbox': (969.2038269042969, 1246.6363220214844, 979.62646484375, 1256.4152526855469), 'score': 0.4577777087688446}, {'type': 'car', 'bbox': (863.8691558837891, 1219.696044921875, 879.1142578125, 1234.1046142578125), 'score': 0.3527847230434418}, {'type': 'car', 'bbox': (825.5778503417969, 1204.5673522949219, 836.7480926513672, 1218.4763793945312), 'score': 0.2990882098674774}, {'type': 'car', 'bbox': (935.1903686523438, 1139.50390625, 943.86083984375, 1148.8598327636719), 'score': 0.25167518854141235}, {'type': 'car', 'bbox': (372.4172668457031, 1656.3837585449219, 379.5776062011719, 1664.1235961914062), 'score': 0.4980125427246094}, {'type': 'car', 'bbox': (168.0048065185547, 1564.6305541992188, 176.23312377929688, 1571.7033386230469), 'score': 0.4345549941062927}, {'type': 'car', 'bbox': (108.9936294555664, 1413.0019989013672, 117.46886444091797, 1421.9583282470703), 'score': 0.37678658962249756}, {'type': 'car', 'bbox': (237.1513671875, 1550.0440063476562, 243.52415466308594, 1559.4335632324219), 'score': 0.2951947748661041}, {'type': 'car', 'bbox': (689.4000434875488, 1545.4311828613281, 696.8457183837891, 1556.5728759765625), 'score': 0.5013344287872314}, {'type': 'car', 'bbox': (1343.8613967895508, 1534.6395568847656, 1352.001853942871, 1541.7393188476562), 'score': 0.41611042618751526}, {'type': 'car', 'bbox': (1448.624496459961, 1633.4820251464844, 1457.5841522216797, 1643.0789184570312), 'score': 0.2849806249141693}] """
drone_metadata = """  """

evaluate_segmentation(fire_mask_data, smoke_mask_data, detected_objects_data, drone_metadata)