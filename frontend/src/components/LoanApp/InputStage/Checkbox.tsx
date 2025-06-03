import {Dispatch, FC, SetStateAction} from "react";

type CheckboxProps = {
	label: string,
	setState: Dispatch<SetStateAction<boolean>>,
	state: boolean
}

export const Checkbox: FC<CheckboxProps> = ({label, setState, state}) => {
	return <div className={"flex flex-col gap-1.5"}>
		<p>{label}:</p>
		<div className={"w-full h-full flex justify-start items-start flex-row"}>
			<input
				type={"checkbox"}
				className={"ml-1 border-black border shadow-xl rounded-md aspect-square h-full bg-white"}
				checked={state}
				onChange={(event) => {
					const element = event.target as HTMLInputElement
					setState(element.checked)
				}}
			/>
		</div>
	</div>
}