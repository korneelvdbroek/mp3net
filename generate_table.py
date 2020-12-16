
file_names = [['fake_sample_334000_0', 'fake_sample_334000_1', 'fake_sample_378000_0', 'fake_sample_378000_1', 'fake_sample_382500_0'],
              ['fake_sample_334000_0', 'fake_sample_334000_1', 'fake_sample_378000_0', 'fake_sample_378000_1', 'fake_sample_382500_0'],
              ['real_sample_00', 'real_sample_01', 'real_sample_02', 'real_sample_03', 'real_sample_04'],
              ['fake_sample_087000_0', 'fake_sample_125000_0', 'fake_sample_243000_0', 'fake_sample_330000_0', 'fake_sample_334000_0']]

text = [['leitmotiv 1', 'leitmotiv 2', 'musical phrase 1', 'musical phrase 2', 'musical phrase 3'],
        ['sample 1', 'sample 2', 'sample 3', 'sample 4', 'sample 5'],
        ['real sample 1', 'real sample 2', 'real sample 3', 'real sample 4', 'real sample 5'],
        ['sample day 1', 'sample day 2', 'sample day 3', 'sample day 4', 'sample day 5'],]

print(f"<table  class='sample-selection'>")
print(f"    <tr>")
print(f"        <td><b>Selected generated samples</b></td>")
print(f"        <td><b>Uncurated generated samples</b></td>")
print(f"        <td><b>Real samples</b></td>")
print(f"        <td><b>Generated samples during training</b></td>")
print(f"    </tr>")
for row in range(len(file_names[0])):
  print(f"    <tr>")
  for col in range(len(file_names)):
    print(f"        <td>")
    print(f"            <label>")
    print(f"                <input type='radio' name='selectSample' value='{file_names[col][row]}' {'checked ' if row == col == 0 else ''}/>")
    print(f"                {text[col][row]}")
    print(f"            </label>")
    print(f"        </td>")
  print(f"    </tr>")
print(f"</table>")
